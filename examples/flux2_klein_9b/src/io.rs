use std::fs::File;
use std::io::{BufReader, BufWriter};

pub fn load_png(path: &str) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    let mut reader = decoder.read_info()?;
    let buf_size = reader
        .output_buffer_size()
        .ok_or("load_png: image too large")?;
    let mut buf = vec![0u8; buf_size];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("load_png: unsupported color type {other:?}").into()),
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!("load_png: unsupported bit depth {:?}", info.bit_depth).into());
    }
    let bytes = &buf[..info.buffer_size()];
    let mut chw = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let byte = bytes[(y * w + x) * channels + c] as f32;
                chw[c * h * w + y * w + x] = byte / 255.0 * 2.0 - 1.0;
            }
        }
    }
    Ok((chw, w, h))
}

pub fn save_png(
    path: &str,
    chw: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(chw.len(), 3 * h * w, "save_png: shape mismatch");
    let mut bytes = vec![0u8; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = chw[c * h * w + y * w + x];
                let v = ((v + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
                bytes[(y * w + x) * 3 + c] = v as u8;
            }
        }
    }
    let file = File::create(path)?;
    let bw = BufWriter::new(file);
    let mut encoder = png::Encoder::new(bw, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&bytes)?;
    Ok(())
}