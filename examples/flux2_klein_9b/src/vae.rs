use luminal::{graph::Graph, prelude::*};

use crate::util::Materialize;

/// Standard AutoencoderKL constants for Flux 2.
pub const LATENT_CHANNELS: usize = 32;
/// The encoder emits `2 * LATENT_CHANNELS` channels: the first half is the
/// distribution mean, the second the log-variance (a `DiagonalGaussian`).
pub const MOMENT_CHANNELS: usize = 2 * LATENT_CHANNELS;
pub const VAE_DOWNSAMPLE: usize = 8; // 3 spatial halvings on the encoder side.
pub const NORM_NUM_GROUPS: usize = 32;
pub const NORM_EPS: f32 = 1e-6;
pub const BLOCK_OUT_CHANNELS: [usize; 4] = [128, 256, 512, 512];
pub const LAYERS_PER_BLOCK: usize = 2; // diffusers config; the decoder uses 3 resnets/block (= layers_per_block + 1).
pub const RESNETS_PER_BLOCK: usize = LAYERS_PER_BLOCK + 1;

fn conv2d_bias(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> GraphTensor {
    let dims = x.dims();
    assert_eq!(dims.len(), 3, "conv2d_bias expects (C, H, W)");
    let h = dims[1];
    let w = dims[2];

    if kernel == 1 && stride == 1 && padding == 0 {
        let xt = x.permute(&[1, 2, 0]).merge_dims(0, 1); // (H*W, C_in)
        let out = xt.matmul(weight.t()); // (H*W, C_out)
        let out = out.split_dims(0, w).permute(&[2, 0, 1]); // (C_out, H, W)
        return out + bias.expand_dim(1, h).expand_dim(2, w);
    }

    let zero = Expression::from(0);
    let pad = Expression::from(padding);
    let padded = if padding > 0 {
        x.pad(vec![(zero, zero), (pad, pad), (pad, pad)], 0.0)
    } else {
        x
    };

    let unfolded = padded.unfold(
        vec![1usize, kernel, kernel],
        vec![1usize, stride, stride],
        vec![1usize, 1, 1],
    );
    let output_spatial_dims = unfolded.dims()[1..3].to_vec();

    // (C, H_out, W_out, 1, K, K) -> (H_out, W_out, C, K, K)
    let mut patches = unfolded.squeeze(3).permute(&[1, 2, 0, 3, 4]);
    while patches.dims().len() > 3 {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    let patches = patches.merge_dims(0, 1).materialize(); // (H_out*W_out, C_in*K*K)

    let out = patches.matmul(weight.t().materialize()); // (H_out*W_out, C_out)
    let out = out
        .split_dims(0, output_spatial_dims[1])
        .permute(&[2, 0, 1]); // (C_out, H_out, W_out)
    let out_dims = out.dims();
    out + bias.expand_dim(1, out_dims[1]).expand_dim(2, out_dims[2])
}

fn linear_bias(x: GraphTensor, weight: GraphTensor, bias: GraphTensor) -> GraphTensor {
    let out = x.matmul(weight.cast(x.dtype).t());
    let out_dims = out.dims();
    match out_dims.len() {
        1 => out + bias,
        2 => out + bias.expand_dim(0, out_dims[0]),
        3 => out + bias.expand_dim(0, out_dims[0]).expand_dim(1, out_dims[1]),
        n => panic!("linear_bias: unsupported rank {n}"),
    }
}

fn group_norm(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    num_groups: usize,
    eps: f32,
) -> GraphTensor {
    let dims = x.dims();
    assert_eq!(dims.len(), 3, "group_norm expects (C, H, W)");
    let c = dims[0];
    let h = dims[1];
    let w = dims[2];

    let c_const = c
        .to_usize()
        .expect("num_channels must be static for GroupNorm");
    let h_const = h.to_usize().expect("height must be static for GroupNorm");
    let w_const = w.to_usize().expect("width must be static for GroupNorm");
    assert!(
        c_const.is_multiple_of(num_groups),
        "num_channels ({c_const}) must be a multiple of num_groups ({num_groups})",
    );
    let group_size = c_const / num_groups;
    let group_volume = group_size * h_const * w_const;

    let flat = x.merge_dims(0, 1).merge_dims(0, 1); // (C*H*W,)
    let grouped = flat.split_dims(0, group_volume); // (num_groups, group_volume)

    let normed = grouped.layer_norm(1, eps);

    let unshaped = normed
        .merge_dims(0, 1) // flat (C*H*W,)
        .split_dims(0, h_const * w_const) // (C, H*W)
        .split_dims(1, w_const); // (C, H, W)

    let w_b = weight.expand_dim(1, h).expand_dim(2, w);
    let b_b = bias.expand_dim(1, h).expand_dim(2, w);
    unshaped * w_b + b_b
}

fn nearest_upsample_2x(x: GraphTensor) -> GraphTensor {
    // (C, H, W) -> (C, H, 2, W) -> (C, 2H, W) -> (C, 2H, W, 2) -> (C, 2H, 2W)
    let stage1 = x.expand_dim(2, 2_usize).merge_dims(1, 2);
    stage1.expand_dim(3, 2_usize).merge_dims(2, 3)
}

fn silu(x: GraphTensor) -> GraphTensor {
    x.silu()
}

struct ResnetBlock {
    norm1_w: GraphTensor,
    norm1_b: GraphTensor,
    conv1_w: GraphTensor,
    conv1_b: GraphTensor,
    norm2_w: GraphTensor,
    norm2_b: GraphTensor,
    conv2_w: GraphTensor,
    conv2_b: GraphTensor,
    shortcut: Option<(GraphTensor, GraphTensor)>, // 1×1 conv when in_c != out_c
    in_channels: usize,
    out_channels: usize,
}

impl ResnetBlock {
    fn new(prefix: &str, in_c: usize, out_c: usize, cx: &mut Graph) -> Self {
        let shortcut = if in_c == out_c {
            None
        } else {
            Some((
                cx.named_tensor(format!("{prefix}.conv_shortcut.weight"), (out_c, in_c))
                    .persist(),
                cx.named_tensor(format!("{prefix}.conv_shortcut.bias"), out_c)
                    .persist(),
            ))
        };
        Self {
            norm1_w: cx
                .named_tensor(format!("{prefix}.norm1.weight"), in_c)
                .persist(),
            norm1_b: cx
                .named_tensor(format!("{prefix}.norm1.bias"), in_c)
                .persist(),
            conv1_w: cx
                .named_tensor(format!("{prefix}.conv1.weight"), (out_c, in_c * 3 * 3))
                .persist(),
            conv1_b: cx
                .named_tensor(format!("{prefix}.conv1.bias"), out_c)
                .persist(),
            norm2_w: cx
                .named_tensor(format!("{prefix}.norm2.weight"), out_c)
                .persist(),
            norm2_b: cx
                .named_tensor(format!("{prefix}.norm2.bias"), out_c)
                .persist(),
            conv2_w: cx
                .named_tensor(format!("{prefix}.conv2.weight"), (out_c, out_c * 3 * 3))
                .persist(),
            conv2_b: cx
                .named_tensor(format!("{prefix}.conv2.bias"), out_c)
                .persist(),
            shortcut,
            in_channels: in_c,
            out_channels: out_c,
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let h = group_norm(x, self.norm1_w, self.norm1_b, NORM_NUM_GROUPS, NORM_EPS);
        let h = silu(h);
        let h = conv2d_bias(h, self.conv1_w, self.conv1_b, 3, 1, 1);
        let h = group_norm(h, self.norm2_w, self.norm2_b, NORM_NUM_GROUPS, NORM_EPS);
        let h = silu(h);
        let h = conv2d_bias(h, self.conv2_w, self.conv2_b, 3, 1, 1);

        let skip = if self.in_channels == self.out_channels {
            x
        } else {
            let (sw, sb) = self.shortcut.expect("shortcut required when in_c != out_c");
            conv2d_bias(x, sw, sb, 1, 1, 0)
        };
        skip + h
    }
}

struct AttnBlock {
    group_norm_w: GraphTensor,
    group_norm_b: GraphTensor,
    to_q_w: GraphTensor,
    to_q_b: GraphTensor,
    to_k_w: GraphTensor,
    to_k_b: GraphTensor,
    to_v_w: GraphTensor,
    to_v_b: GraphTensor,
    to_out_w: GraphTensor,
    to_out_b: GraphTensor,
    channels: usize,
}

impl AttnBlock {
    fn new(prefix: &str, channels: usize, cx: &mut Graph) -> Self {
        let lin =
            |name: &str, out: usize, inn: usize, cx: &mut Graph| -> (GraphTensor, GraphTensor) {
                (
                    cx.named_tensor(format!("{prefix}.{name}.weight"), (out, inn))
                        .persist(),
                    cx.named_tensor(format!("{prefix}.{name}.bias"), out)
                        .persist(),
                )
            };
        let (to_q_w, to_q_b) = lin("to_q", channels, channels, cx);
        let (to_k_w, to_k_b) = lin("to_k", channels, channels, cx);
        let (to_v_w, to_v_b) = lin("to_v", channels, channels, cx);
        let (to_out_w, to_out_b) = lin("to_out.0", channels, channels, cx);
        Self {
            group_norm_w: cx
                .named_tensor(format!("{prefix}.group_norm.weight"), channels)
                .persist(),
            group_norm_b: cx
                .named_tensor(format!("{prefix}.group_norm.bias"), channels)
                .persist(),
            to_q_w,
            to_q_b,
            to_k_w,
            to_k_b,
            to_v_w,
            to_v_b,
            to_out_w,
            to_out_b,
            channels,
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let dims = x.dims();
        assert_eq!(dims.len(), 3, "AttnBlock expects (C, H, W)");
        let _h = dims[1];
        let w = dims[2];
        let residual = x;

        // GroupNorm + reshape to (HW, C) for linear projections.
        let normed = group_norm(
            x,
            self.group_norm_w,
            self.group_norm_b,
            NORM_NUM_GROUPS,
            NORM_EPS,
        );
        // (C, H, W) -> (C, H*W) -> (H*W, C). This is a column-major view
        // that cuBLASLt can consume directly.
        let merged = normed.merge_dims(1, 2).transpose(0, 1);

        let q = linear_bias(merged, self.to_q_w, self.to_q_b);
        let k = linear_bias(merged, self.to_k_w, self.to_k_b);
        let v = linear_bias(merged, self.to_v_w, self.to_v_b);

        // Standard scaled dot-product attention over the spatial axis.
        let scale = (self.channels as f32).sqrt().recip();
        let scores = q.matmul(k.t()) * scale;
        let attn_w = scores.softmax(1);
        let attn = attn_w.matmul(v);

        let out = linear_bias(attn, self.to_out_w, self.to_out_b);
        // (H*W, C) -> (C, H*W) -> (C, H, W)
        let out = out.transpose(0, 1).split_dims(1, w);
        residual + out
    }
}

struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsampler: Option<(GraphTensor, GraphTensor)>, // 3×3 conv after nearest-2×
}

impl UpBlock {
    fn new(prefix: &str, in_c: usize, out_c: usize, with_upsampler: bool, cx: &mut Graph) -> Self {
        let mut resnets = Vec::with_capacity(RESNETS_PER_BLOCK);
        for r in 0..RESNETS_PER_BLOCK {
            let resnet_in = if r == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::new(
                &format!("{prefix}.resnets.{r}"),
                resnet_in,
                out_c,
                cx,
            ));
        }
        let upsampler = if with_upsampler {
            Some((
                cx.named_tensor(
                    format!("{prefix}.upsamplers.0.conv.weight"),
                    (out_c, out_c * 3 * 3),
                )
                .persist(),
                cx.named_tensor(format!("{prefix}.upsamplers.0.conv.bias"), out_c)
                    .persist(),
            ))
        } else {
            None
        };
        Self { resnets, upsampler }
    }

    fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        for r in &self.resnets {
            x = r.forward(x);
        }
        if let Some((w, b)) = &self.upsampler {
            let up = nearest_upsample_2x(x);
            x = conv2d_bias(up, *w, *b, 3, 1, 1);
        }
        x
    }
}

fn decoder_block_channels(block_idx: usize) -> (usize, usize) {
    let n = BLOCK_OUT_CHANNELS.len();
    let reversed = n - 1 - block_idx;
    let prev = if reversed + 1 < n {
        BLOCK_OUT_CHANNELS[reversed + 1]
    } else {
        BLOCK_OUT_CHANNELS[reversed]
    };
    let out = BLOCK_OUT_CHANNELS[reversed];
    let in_c = if block_idx == 0 {
        BLOCK_OUT_CHANNELS[n - 1] // mid block runs at the deepest channel count
    } else {
        prev
    };
    (in_c, out)
}

pub struct VaeDecoder {
    post_quant_w: GraphTensor,
    post_quant_b: GraphTensor,
    conv_in_w: GraphTensor,
    conv_in_b: GraphTensor,
    mid_resnet_0: ResnetBlock,
    mid_attn: AttnBlock,
    mid_resnet_1: ResnetBlock,
    up_blocks: Vec<UpBlock>,
    norm_out_w: GraphTensor,
    norm_out_b: GraphTensor,
    conv_out_w: GraphTensor,
    conv_out_b: GraphTensor,
}

impl VaeDecoder {
    pub fn new(cx: &mut Graph) -> Self {
        let post_quant_w = cx
            .named_tensor("post_quant_conv.weight", (LATENT_CHANNELS, LATENT_CHANNELS))
            .persist();
        let post_quant_b = cx
            .named_tensor("post_quant_conv.bias", LATENT_CHANNELS)
            .persist();

        let mid = BLOCK_OUT_CHANNELS[BLOCK_OUT_CHANNELS.len() - 1];
        let conv_in_w = cx
            .named_tensor("decoder.conv_in.weight", (mid, LATENT_CHANNELS * 3 * 3))
            .persist();
        let conv_in_b = cx.named_tensor("decoder.conv_in.bias", mid).persist();

        let mid_resnet_0 = ResnetBlock::new("decoder.mid_block.resnets.0", mid, mid, cx);
        let mid_attn = AttnBlock::new("decoder.mid_block.attentions.0", mid, cx);
        let mid_resnet_1 = ResnetBlock::new("decoder.mid_block.resnets.1", mid, mid, cx);

        let mut up_blocks = Vec::with_capacity(BLOCK_OUT_CHANNELS.len());
        for b in 0..BLOCK_OUT_CHANNELS.len() {
            let (in_c, out_c) = decoder_block_channels(b);
            let with_upsampler = b < BLOCK_OUT_CHANNELS.len() - 1;
            up_blocks.push(UpBlock::new(
                &format!("decoder.up_blocks.{b}"),
                in_c,
                out_c,
                with_upsampler,
                cx,
            ));
        }

        let last_c = BLOCK_OUT_CHANNELS[0];
        let norm_out_w = cx
            .named_tensor("decoder.conv_norm_out.weight", last_c)
            .persist();
        let norm_out_b = cx
            .named_tensor("decoder.conv_norm_out.bias", last_c)
            .persist();
        let conv_out_w = cx
            .named_tensor("decoder.conv_out.weight", (3, last_c * 3 * 3))
            .persist();
        let conv_out_b = cx.named_tensor("decoder.conv_out.bias", 3).persist();

        Self {
            post_quant_w,
            post_quant_b,
            conv_in_w,
            conv_in_b,
            mid_resnet_0,
            mid_attn,
            mid_resnet_1,
            up_blocks,
            norm_out_w,
            norm_out_b,
            conv_out_w,
            conv_out_b,
        }
    }

    /// Decode a latent of shape (LATENT_CHANNELS, h, w) into an RGB image
    /// of shape (3, h * VAE_DOWNSAMPLE, w * VAE_DOWNSAMPLE) in the [-1, 1] range.
    pub fn forward(&self, latent: GraphTensor) -> GraphTensor {
        self.forward_partial(latent, usize::MAX)
    }

    /// Run the decoder up to stage `stop_at` (used for incremental debugging).
    /// Stages: 0=post_quant only, 1=+conv_in, 2..=4=+mid (resnet, attn, resnet),
    /// 5..=8=+up_blocks[0..3], 9=+conv_norm_out+silu, 10=+conv_out (full).
    pub fn forward_partial(&self, latent: GraphTensor, stop_at: usize) -> GraphTensor {
        let mut x = conv2d_bias(latent, self.post_quant_w, self.post_quant_b, 1, 1, 0);
        if stop_at == 0 {
            return x;
        }
        x = conv2d_bias(x, self.conv_in_w, self.conv_in_b, 3, 1, 1);
        if stop_at == 1 {
            return x;
        }
        x = self.mid_resnet_0.forward(x);
        if stop_at == 2 {
            return x;
        }
        x = self.mid_attn.forward(x);
        if stop_at == 3 {
            return x;
        }
        x = self.mid_resnet_1.forward(x);
        if stop_at == 4 {
            return x;
        }
        for (i, blk) in self.up_blocks.iter().enumerate() {
            x = blk.forward(x);
            if stop_at == 5 + i {
                return x;
            }
        }
        x = group_norm(
            x,
            self.norm_out_w,
            self.norm_out_b,
            NORM_NUM_GROUPS,
            NORM_EPS,
        );
        x = silu(x);
        if stop_at == 9 {
            return x;
        }
        conv2d_bias(x, self.conv_out_w, self.conv_out_b, 3, 1, 1)
    }
}

fn downsample_conv(x: GraphTensor, weight: GraphTensor, bias: GraphTensor) -> GraphTensor {
    let zero = Expression::from(0);
    let one = Expression::from(1);
    let padded = x.pad(vec![(zero, zero), (zero, one), (zero, one)], 0.0);
    conv2d_bias(padded, weight, bias, 3, 2, 0)
}

struct DownBlock {
    resnets: Vec<ResnetBlock>,
    downsampler: Option<(GraphTensor, GraphTensor)>,
}

impl DownBlock {
    fn new(prefix: &str, in_c: usize, out_c: usize, with_downsampler: bool, cx: &mut Graph) -> Self {
        // The encoder runs `layers_per_block` resnets per down block (2), unlike
        // the decoder's `layers_per_block + 1` (3).
        let mut resnets = Vec::with_capacity(LAYERS_PER_BLOCK);
        for r in 0..LAYERS_PER_BLOCK {
            let resnet_in = if r == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::new(
                &format!("{prefix}.resnets.{r}"),
                resnet_in,
                out_c,
                cx,
            ));
        }
        let downsampler = if with_downsampler {
            Some((
                cx.named_tensor(
                    format!("{prefix}.downsamplers.0.conv.weight"),
                    (out_c, out_c * 3 * 3),
                )
                .persist(),
                cx.named_tensor(
                    format!("{prefix}.downsamplers.0.conv.bias"),
                    out_c
                )
                .persist(),
            ))
        } else {
            None
        };
        Self {
            resnets,
            downsampler,
        }
    }

    fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        for r in &self.resnets {
            x = r.forward(x);
        }
        if let Some((w, b)) = &self.downsampler {
            x = downsample_conv(x, *w, *b);
        }
        x
    }
}

fn encoder_block_channels(block_idx: usize) -> (usize, usize) {
    let out = BLOCK_OUT_CHANNELS[block_idx];
    let in_c = if block_idx == 0 {
        BLOCK_OUT_CHANNELS[0]
    } else {
        BLOCK_OUT_CHANNELS[block_idx - 1]
    };
    (in_c, out)
}

pub struct VaeEncoder {
    conv_in_w: GraphTensor,
    conv_in_b: GraphTensor,
    down_blocks: Vec<DownBlock>,
    mid_resnet_0: ResnetBlock,
    mid_attn: AttnBlock,
    mid_resnet_1: ResnetBlock,
    norm_out_w: GraphTensor,
    norm_out_b: GraphTensor,
    conv_out_w: GraphTensor,
    conv_out_b: GraphTensor,
    // 1×1 `quant_conv` mapping the `2*LATENT_CHANNELS` moments to themselves.
    quant_w: GraphTensor,
    quant_b: GraphTensor,
}

impl VaeEncoder {
    pub fn new(cx: &mut Graph) -> Self {
        let first_c = BLOCK_OUT_CHANNELS[0];
        let conv_in_w = cx
            .named_tensor("encoder.conv_in.weight", (first_c, 3 * 3 * 3))
            .persist();
        let conv_in_b = cx.named_tensor("encoder.conv_in.bias", first_c).persist();

        let mut down_blocks = Vec::with_capacity(BLOCK_OUT_CHANNELS.len());
        for b in 0..BLOCK_OUT_CHANNELS.len() {
            let (in_c, out_c) = encoder_block_channels(b);
            // Every block downsamples except the deepest one.
            let with_downsampler = b < BLOCK_OUT_CHANNELS.len() - 1;
            down_blocks.push(DownBlock::new(
                &format!("encoder.down_blocks.{b}"),
                in_c,
                out_c,
                with_downsampler,
                cx,
            ));
        }

        let mid = BLOCK_OUT_CHANNELS[BLOCK_OUT_CHANNELS.len() - 1];
        let mid_resnet_0 = ResnetBlock::new("encoder.mid_block.resnets.0", mid, mid, cx);
        let mid_attn = AttnBlock::new("encoder.mid_block.attentions.0", mid, cx);
        let mid_resnet_1 = ResnetBlock::new("encoder.mid_block.resnets.1", mid, mid, cx);

        let norm_out_w = cx
            .named_tensor("encoder.conv_norm_out.weight", mid)
            .persist();
        let norm_out_b = cx.named_tensor("encoder.conv_norm_out.bias", mid).persist();
        let conv_out_w = cx
            .named_tensor("encoder.conv_out.weight", (MOMENT_CHANNELS, mid * 3 * 3))
            .persist();
        let conv_out_b = cx
            .named_tensor("encoder.conv_out.bias", MOMENT_CHANNELS)
            .persist();

        let quant_w = cx
            .named_tensor("quant_conv.weight", (MOMENT_CHANNELS, MOMENT_CHANNELS))
            .persist();
        let quant_b = cx.named_tensor("quant_conv.bias", MOMENT_CHANNELS).persist();

        Self {
            conv_in_w,
            conv_in_b,
            down_blocks,
            mid_resnet_0,
            mid_attn,
            mid_resnet_1,
            norm_out_w,
            norm_out_b,
            conv_out_w,
            conv_out_b,
            quant_w,
            quant_b,
        }
    }

    /// Encode an RGB image `(3, h, w)` in the [-1, 1] range into the Gaussian
    /// moments `(MOMENT_CHANNELS, h/VAE_DOWNSAMPLE, w/VAE_DOWNSAMPLE)`: channels
    /// `0..LATENT_CHANNELS` are the mean, `LATENT_CHANNELS..` the log-variance.
    pub fn forward(&self, image: GraphTensor) -> GraphTensor {
        let mut x = conv2d_bias(image, self.conv_in_w, self.conv_in_b, 3, 1, 1);
        for blk in &self.down_blocks {
            x = blk.forward(x);
        }
        x = self.mid_resnet_0.forward(x);
        x = self.mid_attn.forward(x);
        x = self.mid_resnet_1.forward(x);
        x = group_norm(x, self.norm_out_w, self.norm_out_b, NORM_NUM_GROUPS, NORM_EPS);
        x = silu(x);
        x = conv2d_bias(x, self.conv_out_w, self.conv_out_b, 3, 1, 1);
        conv2d_bias(x, self.quant_w, self.quant_b, 1, 1, 0)
    }

    /// Encode an image and return the distribution mean
    /// `(LATENT_CHANNELS, h/VAE_DOWNSAMPLE, w/VAE_DOWNSAMPLE)` — the deterministic
    /// latent used for reconstruction / img2img.
    pub fn encode_mean(&self, image: GraphTensor) -> GraphTensor {
        // Take the mean half of the moments. The first LATENT_CHANNELS channels
        // are a contiguous prefix of the 64-channel buffer, so the slice stays
        // `is_contiguous()` and would otherwise be dropped (leaving all 64
        // channels, incl. the large logvar half). `materialize()` forces a copy
        // of just the 32 mean channels.
        self.forward(image).slice_along(..LATENT_CHANNELS, 0).materialize()
    }
}

#[cfg(test)]
mod tests {
    use luminal::hlir::CustomOpKind;
    use luminal_rocm_lite::{rocmrc::hip::HipContext, runtime::RocmRuntime};

    use super::*;

    fn assert_no_custom_ops(cx: &Graph) {
        assert!(
            cx.custom_ops.is_empty(),
            "Flux2 VAE helpers should use pure HLIR, not registered CustomOp wrappers"
        );
        let custom_nodes: Vec<_> = cx
            .graph
            .node_indices()
            .filter(|&node| cx.try_get_op::<CustomOpKind>(node).is_some())
            .collect();
        assert!(
            custom_nodes.is_empty(),
            "Flux2 VAE graph contains CustomOpKind nodes: {custom_nodes:?}"
        );
    }

    #[test]
    fn vae_helpers_use_no_custom_ops() {
        let mut cx = Graph::default();

        let x = cx.named_tensor("x", (2usize, 3usize, 3usize));
        let conv_w = cx.named_tensor("conv_w", (4usize, 2usize * 3 * 3));
        let conv_b = cx.named_tensor("conv_b", 4usize);
        let _ = conv2d_bias(x, conv_w, conv_b, 3, 1, 1).output();

        let lin_x = cx.named_tensor("lin_x", (2usize, 3usize));
        let lin_w = cx.named_tensor("lin_w", (4usize, 3usize));
        let lin_b = cx.named_tensor("lin_b", 4usize);
        let _ = linear_bias(lin_x, lin_w, lin_b).output();

        assert_no_custom_ops(&cx);
    }

    struct Conv2dCase {
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    }

    fn reference_conv2d_bias(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        case: Conv2dCase,
    ) -> Vec<f32> {
        let Conv2dCase {
            c_in,
            h,
            w,
            c_out,
            kernel,
            stride,
            padding,
        } = case;
        let h_out = (h + 2 * padding - kernel) / stride + 1;
        let w_out = (w + 2 * padding - kernel) / stride + 1;
        let mut out = vec![0.0_f32; c_out * h_out * w_out];
        for co in 0..c_out {
            for oy in 0..h_out {
                for ox in 0..w_out {
                    let mut acc = bias[co];
                    for ci in 0..c_in {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                let iy_padded = oy * stride + ky;
                                let ix_padded = ox * stride + kx;
                                if iy_padded < padding || ix_padded < padding {
                                    continue;
                                }
                                let iy = iy_padded - padding;
                                let ix = ix_padded - padding;
                                if iy >= h || ix >= w {
                                    continue;
                                }
                                let input_idx = ci * h * w + iy * w + ix;
                                let weight_idx = co * c_in * kernel * kernel
                                    + ci * kernel * kernel
                                    + ky * kernel
                                    + kx;
                                acc += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                    out[co * h_out * w_out + oy * w_out + ox] = acc;
                }
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*a - *e).abs() < 1e-4,
                "value mismatch at {idx}: got {a}, expected {e}"
            );
        }
    }

    fn reference_nearest_upsample_2x(input: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; c * h * 2 * w * 2];
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let value = input[ci * h * w + y * w + x];
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let oy = y * 2 + dy;
                            let ox = x * 2 + dx;
                            out[ci * h * 2 * w * 2 + oy * w * 2 + ox] = value;
                        }
                    }
                }
            }
        }
        out
    }

    struct GroupNormCase {
        c: usize,
        h: usize,
        w: usize,
        num_groups: usize,
        eps: f32,
    }

    fn reference_group_norm(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        case: GroupNormCase,
    ) -> Vec<f32> {
        let GroupNormCase {
            c,
            h,
            w,
            num_groups,
            eps,
        } = case;
        let group_size = c / num_groups;
        let group_volume = group_size * h * w;
        let mut out = vec![0.0_f32; input.len()];
        for group in 0..num_groups {
            let c_start = group * group_size;
            let mut mean = 0.0_f32;
            for ci in c_start..c_start + group_size {
                for idx in 0..h * w {
                    mean += input[ci * h * w + idx];
                }
            }
            mean /= group_volume as f32;

            let mut variance = 0.0_f32;
            for ci in c_start..c_start + group_size {
                for idx in 0..h * w {
                    let centered = input[ci * h * w + idx] - mean;
                    variance += centered * centered;
                }
            }
            variance /= group_volume as f32;
            let inv_std = (variance + eps).sqrt().recip();

            for ci in c_start..c_start + group_size {
                for idx in 0..h * w {
                    let flat = ci * h * w + idx;
                    out[flat] = (input[flat] - mean) * inv_std * weight[ci] + bias[ci];
                }
            }
        }
        out
    }

    fn one_search() -> CompileOptions {
        CompileOptions::default().search_graph_limit(1)
    }

    #[test]
    fn conv2d_bias_matches_reference() {
        let mut cx = Graph::default();
        let input_t = cx.named_tensor("input", (2usize, 3usize, 3usize));
        let weight_t = cx.named_tensor("weight", (2usize, 2usize * 3 * 3));
        let bias_t = cx.named_tensor("bias", 2usize);
        let out = conv2d_bias(input_t, weight_t, bias_t, 3, 1, 1).output();

        let input: Vec<f32> = (0..18).map(|i| i as f32 * 0.1 - 0.7).collect();
        let weight: Vec<f32> = (0..36).map(|i| (i as f32 % 7.0) * 0.05 - 0.15).collect();
        let bias = vec![0.25_f32, -0.5_f32];
        let expected = reference_conv2d_bias(
            &input,
            &weight,
            &bias,
            Conv2dCase {
                c_in: 2,
                h: 3,
                w: 3,
                c_out: 2,
                kernel: 3,
                stride: 1,
                padding: 1,
            },
        );

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(ReferenceRuntime::default(), one_search());
        rt.set_data(input_t, input);
        rt.set_data(weight_t, weight);
        rt.set_data(bias_t, bias);
        rt.execute(&cx.dyn_map);

        assert_close(rt.get_f32(out.id), &expected);
    }

    /// Deep residual chain [group_norm → silu → conv → +residual] × N, compared
    /// against a HOST reference (ground truth) — not the CPU runtime, which
    /// shares the same egglog saturation and would be equally wrong if an op is
    /// dropped. If the GPU diverges from the host, egglog dropped an op.
    #[test]
    fn deep_resnet_chain_matches_host_reference() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();
        let (c, h, w, groups, n) = (64usize, 16usize, 16usize, 32usize, 6usize);

        let mut x_v: Vec<f32> = (0..c * h * w).map(|i| ((i % 101) as f32 - 50.0) * 0.05).collect();
        let gnw_v = vec![0.9_f32; c];
        let gnb_v = vec![0.02_f32; c];
        let cw_v: Vec<f32> = (0..c * c * 9).map(|i| ((i % 17) as f32 - 8.0) * 0.004).collect();
        let cb_v = vec![0.0_f32; c];

        // Host reference: same [gn → silu → conv → +residual] × n.
        let host_silu = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| x / (1.0 + (-x).exp())).collect() };
        let mut expected = x_v.clone();
        for _ in 0..n {
            let gn = reference_group_norm(
                &expected,
                &gnw_v,
                &gnb_v,
                GroupNormCase { c, h, w, num_groups: groups, eps: 1e-6 },
            );
            let s = host_silu(&gn);
            let hh = reference_conv2d_bias(
                &s,
                &cw_v,
                &cb_v,
                Conv2dCase { c_in: c, h, w, c_out: c, kernel: 3, stride: 1, padding: 1 },
            );
            for (e, d) in expected.iter_mut().zip(&hh) {
                *e += d;
            }
        }

        // GPU graph.
        let mut cx = Graph::default();
        let x_t = cx.named_tensor("x", (c, h, w));
        let gnw = cx.named_tensor("gnw", c);
        let gnb = cx.named_tensor("gnb", c);
        let cw = cx.named_tensor("cw", (c, c * 9));
        let cb = cx.named_tensor("cb", c);
        let mut x = x_t;
        for _ in 0..n {
            let hh = conv2d_bias(silu(group_norm(x, gnw, gnb, groups, 1e-6)), cw, cb, 3, 1, 1);
            x = x + hh;
        }
        let out = x.output();
        cx.build_search_space::<RocmRuntime>(CompileOptions::default());
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.set_data(x_t, std::mem::take(&mut x_v));
        rt.set_data(gnw, gnw_v);
        rt.set_data(gnb, gnb_v);
        rt.set_data(cw, cw_v);
        rt.set_data(cb, cb_v);
        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);
        let gpu = rt.get_f32(out.id);

        let std = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        println!("deep chain: host std={:.4} gpu std={:.4}", std(&expected), std(&gpu));
        for (i, (g, e)) in gpu.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() < 1e-2, "GPU≠host at {i}: gpu={g}, host={e}");
        }
    }

    /// Full downsampler (asymmetric `pad(0,1,0,1)` → 3×3 stride-2 conv) vs a
    /// HOST reference. `down_block_0` grows with just a downsampler (no
    /// shortcut), and `pad` builds a masked border view — if the zero-mask
    /// isn't materialized, the border reads garbage and the output blows up.
    #[test]
    fn downsample_conv_matches_host_reference() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();
        let (c, h, w) = (64usize, 16usize, 16usize);

        let x: Vec<f32> = (0..c * h * w).map(|i| ((i % 101) as f32 - 50.0) * 0.05).collect();
        let wv: Vec<f32> = (0..c * c * 9).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let bv = vec![0.0_f32; c];

        // Host reference: pad bottom/right by 1 (zeros), then 3×3 stride-2 pad-0.
        let (hp, wp) = (h + 1, w + 1);
        let mut padded = vec![0.0_f32; c * hp * wp];
        for ci in 0..c {
            for y in 0..h {
                for xx in 0..w {
                    padded[ci * hp * wp + y * wp + xx] = x[ci * h * w + y * w + xx];
                }
            }
        }
        let expected = reference_conv2d_bias(
            &padded,
            &wv,
            &bv,
            Conv2dCase { c_in: c, h: hp, w: wp, c_out: c, kernel: 3, stride: 2, padding: 0 },
        );

        let mut cx = Graph::default();
        let xt = cx.named_tensor("x", (c, h, w));
        let wt = cx.named_tensor("w", (c, c * 9));
        let bt = cx.named_tensor("b", c);
        let out = downsample_conv(xt, wt, bt).output();
        cx.build_search_space::<RocmRuntime>(CompileOptions::default());
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.set_data(xt, x);
        rt.set_data(wt, wv);
        rt.set_data(bt, bv);
        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);
        let gpu = rt.get_f32(out.id);

        let std = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        println!("downsample: host std={:.4} gpu std={:.4} (len host={} gpu={})", std(&expected), std(&gpu), expected.len(), gpu.len());
        assert_eq!(gpu.len(), expected.len(), "length mismatch");
        for (i, (g, e)) in gpu.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() < 1e-2, "GPU≠host at {i}: gpu={g}, host={e}");
        }
    }

    /// GPU conv at the downsampler regime: 3×3, **stride 2** (only stride-1 was
    /// tested). The encoder's growth correlates with the downsamplers.
    #[test]
    fn conv2d_bias_stride2_matches_reference_rocm() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();
        let (c_in, c_out, h, w) = (64usize, 64usize, 16usize, 16usize);

        let mut cx = Graph::default();
        let input_t = cx.named_tensor("input", (c_in, h, w));
        let weight_t = cx.named_tensor("weight", (c_out, c_in * 3 * 3));
        let bias_t = cx.named_tensor("bias", c_out);
        let out = conv2d_bias(input_t, weight_t, bias_t, 3, 2, 0).output();

        let input: Vec<f32> = (0..c_in * h * w).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let weight: Vec<f32> = (0..c_out * c_in * 9).map(|i| (i % 7) as f32 * 0.02 - 0.06).collect();
        let bias: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.01).collect();
        let expected = reference_conv2d_bias(
            &input,
            &weight,
            &bias,
            Conv2dCase { c_in, h, w, c_out, kernel: 3, stride: 2, padding: 0 },
        );

        cx.build_search_space::<RocmRuntime>(CompileOptions::default());
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.set_data(input_t, input);
        rt.set_data(weight_t, weight);
        rt.set_data(bias_t, bias);
        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);

        let got = rt.get_f32(out.id);
        assert_eq!(got.len(), expected.len(), "output length mismatch");
        for (i, (a, e)) in got.iter().zip(&expected).enumerate() {
            assert!((a - e).abs() < 1e-2, "mismatch at {i}: got {a}, expected {e}");
        }
    }

    #[test]
    fn nearest_upsample_2x_matches_reference_runtime() {
        let mut cx = Graph::default();
        let input_t = cx.named_tensor("input", (2usize, 3usize, 4usize));
        let out = nearest_upsample_2x(input_t).output();

        let input: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 - 11.0).collect();
        let expected = reference_nearest_upsample_2x(&input, 2, 3, 4);

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(ReferenceRuntime::default(), one_search());
        rt.set_data(input_t, input);
        rt.execute(&cx.dyn_map);

        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn nearest_upsample_2x_matches_reference_rocm() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();

        let mut cx = Graph::default();
        let input_t = cx.named_tensor("input", (2usize, 3usize, 4usize));
        let out = nearest_upsample_2x(input_t).output();

        let input: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 - 11.0).collect();
        let expected = reference_nearest_upsample_2x(&input, 2, 3, 4);

        cx.build_search_space::<RocmRuntime>(CompileOptions::default());
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.set_data(input_t, input);
        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);

        assert_close(&rt.get_f32(out.id), &expected);
    }

    #[test]
    fn group_norm_matches_reference_runtime() {
        let mut cx = Graph::default();
        let input_t = cx.named_tensor("input", (4usize, 2usize, 3usize));
        let weight_t = cx.named_tensor("weight", 4usize);
        let bias_t = cx.named_tensor("bias", 4usize);
        let out = group_norm(input_t, weight_t, bias_t, 2, 1e-6).output();

        let input: Vec<f32> = (0..4 * 2 * 3).map(|i| i as f32 * 0.2 - 2.0).collect();
        let weight = vec![0.7_f32, -0.2, 1.3, 0.5];
        let bias = vec![0.1_f32, -0.3, 0.4, -0.6];
        let expected = reference_group_norm(
            &input,
            &weight,
            &bias,
            GroupNormCase {
                c: 4,
                h: 2,
                w: 3,
                num_groups: 2,
                eps: 1e-6,
            },
        );

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(ReferenceRuntime::default(), one_search());
        rt.set_data(input_t, input);
        rt.set_data(weight_t, weight);
        rt.set_data(bias_t, bias);
        rt.execute(&cx.dyn_map);

        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn group_norm_matches_reference_rocm() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();

        let mut cx = Graph::default();
        let input_t = cx.named_tensor("input", (4usize, 2usize, 3usize));
        let weight_t = cx.named_tensor("weight", 4usize);
        let bias_t = cx.named_tensor("bias", 4usize);
        let out = group_norm(input_t, weight_t, bias_t, 2, 1e-6).output();

        let input: Vec<f32> = (0..4 * 2 * 3).map(|i| i as f32 * 0.2 - 2.0).collect();
        let weight = vec![0.7_f32, -0.2, 1.3, 0.5];
        let bias = vec![0.1_f32, -0.3, 0.4, -0.6];
        let expected = reference_group_norm(
            &input,
            &weight,
            &bias,
            GroupNormCase {
                c: 4,
                h: 2,
                w: 3,
                num_groups: 2,
                eps: 1e-6,
            },
        );

        cx.build_search_space::<RocmRuntime>(CompileOptions::default());
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.set_data(input_t, input);
        rt.set_data(weight_t, weight);
        rt.set_data(bias_t, bias);
        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);

        assert_close(&rt.get_f32(out.id), &expected);
    }

    #[test]
    fn full_vae_decoder_search_rocm() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();

        let h_lat = 512 / VAE_DOWNSAMPLE; // 64
        let w_lat = 512 / VAE_DOWNSAMPLE;

        let mut cx = Graph::default();
        let latent_in = cx.named_tensor("latent", (LATENT_CHANNELS, h_lat, w_lat));
        let decoder = VaeDecoder::new(&mut cx);
        let out = decoder.forward(latent_in).output();
        cx.build_search_space::<RocmRuntime>(CompileOptions::default());

        let Ok(vae_path) = crate::hf::fetch_vae() else {
            // No HF access / weights not cached — skip rather than fail.
            return;
        };
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.load_safetensors(&cx, vae_path.to_str().unwrap());
        rt.set_data(latent_in, vec![0.0_f32; LATENT_CHANNELS * h_lat * w_lat]);

        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);
        let img = rt.get_f32(out.id);
        assert_eq!(img.len(), 3 * 512 * 512);
    }

    #[test]
    fn full_vae_encoder_search_rocm() {
        let Ok(ctx) = HipContext::new(0) else {
            return;
        };
        ctx.bind_to_thread().unwrap();

        let (img_h, img_w) = (512usize, 512usize);
        let h_lat = img_h / VAE_DOWNSAMPLE; // 64
        let w_lat = img_w / VAE_DOWNSAMPLE;

        let mut cx = Graph::default();
        let image_in = cx.named_tensor("image", (3usize, img_h, img_w));
        let encoder = VaeEncoder::new(&mut cx);
        let mean = encoder.encode_mean(image_in).output();
        cx.build_search_space::<RocmRuntime>(CompileOptions::default());

        let Ok(vae_path) = crate::hf::fetch_vae() else {
            // No HF access / weights not cached — skip rather than fail.
            return;
        };
        let mut rt = RocmRuntime::initialize(ctx.default_stream());
        rt.load_safetensors(&cx, vae_path.to_str().unwrap());
        rt.set_data(image_in, vec![0.0_f32; 3 * img_h * img_w]);

        rt = cx.search(rt, one_search());
        rt.execute(&cx.dyn_map);
        let latent = rt.get_f32(mean.id);
        assert_eq!(latent.len(), LATENT_CHANNELS * h_lat * w_lat);
    }
}
