// A JPEG-like coder for grey images, built from your functions: 8 × 8 blocks, DCT, quantise,
// zigzag, symbols. It reports the PSNR and an estimate of the bits (the entropy of the
// symbols plus the value bits: what a Huffman code made for this image would need, roughly).
// Pictures go to out/jpeg/.
//   cargo run -p compression -- jpeg                      (your code)
//   cargo run -p compression --features solution -- jpeg  (the model solution)

use imgcourse::*;
use crate::jpeg::*;
use std::collections::HashMap;

/// Entropy in bits per symbol of a list of symbols.
fn entropy<T: std::hash::Hash + Eq>(symbols: &[T]) -> f64 {
    let mut counts = HashMap::new();
    for s in symbols {
        *counts.entry(s).or_insert(0usize) += 1;
    }
    let n = symbols.len() as f64;
    counts.values().map(|&c| -(c as f64 / n) * (c as f64 / n).log2()).sum()
}

struct Coded {
    decoded: Image<f32>,
    bits: f64,
    nonzero: usize,
}

/// Code an image with grey levels 0..=255 (size a multiple of 8) at a quality.
fn code(img: &Image<f32>, quality: u32) -> Coded {
    let table = scale_table(quality);
    let mut decoded = Image::new(img.width(), img.height(), 0.0);
    let (mut dc_sizes, mut ac) = (Vec::new(), Vec::new());
    let (mut value_bits, mut nonzero, mut previous_dc) = (0u32, 0, 0);
    for by in (0..img.height()).step_by(8) {
        for bx in (0..img.width()).step_by(8) {
            let block = img.crop(bx, by, 8, 8).map(|v| v - 128.0);
            let q = quantise(&dct8(&block), &table);
            nonzero += q.pixels().iter().filter(|&&v| v != 0).count();
            let zz = zigzag(&q);
            // DC: the difference from the previous block, coded as its size + the value bits
            let diff = zz[0] - previous_dc;
            previous_dc = zz[0];
            dc_sizes.push(size_of(diff));
            value_bits += size_of(diff);
            // AC: (run, size) symbols + the value bits
            for (run, size, _) in ac_symbols(&zz) {
                ac.push((run, size));
                value_bits += size;
            }
            let back = idct8(&dequantise(&q, &table));
            for (x, y) in back.coords() {
                decoded[(bx + x, by + y)] = (back[(x, y)] + 128.0).round().clamp(0.0, 255.0);
            }
        }
    }
    let bits = entropy(&dc_sizes) * dc_sizes.len() as f64 + entropy(&ac) * ac.len() as f64 + value_bits as f64;
    Coded { decoded, bits, nonzero }
}

pub fn main() {
    let dir = "out/jpeg";
    let img = samples::cell().map(|v| (v * 255.0).round());
    let n = (img.width() * img.height()) as f64;
    let blocks = n / 64.0;
    save_gray(&img.map(|v| v / 255.0), format!("{dir}/cell.png"));
    println!("quality   bits/pixel   ratio   PSNR (dB)   non-zero values per block");
    for quality in [10, 25, 50, 75, 90] {
        let c = code(&img, quality);
        println!(
            "{quality:>7} {:>12.2} {:>7.1} {:>11.1} {:>15.1}",
            c.bits / n,
            8.0 * n / c.bits,
            psnr(&img, &c.decoded, 255.0),
            c.nonzero as f64 / blocks
        );
        save_gray(&c.decoded.map(|v| v / 255.0), format!("{dir}/q{quality}.png"));
        // the error, × 2, mid grey = 0
        let diff = img.zip_map(&c.decoded, |a, b| 0.5 + 2.0 * (b - a) / 255.0);
        save_gray(&diff, format!("{dir}/q{quality}-error.png"));
    }

    // Generation loss: recompress 10 times at quality 75, moving the image by 4 pixels each
    // time (so the 8 × 8 grid falls elsewhere), or without moving it.
    for (name, step) in [("same grid", 0), ("grid moved", 4)] {
        let mut current = img.clone();
        let mut offset = 0;
        let mut results = Vec::new();
        for _ in 0..10 {
            let shifted = Image::from_fn(256, 256, |x, y| current[((x + step) % 256, (y + step) % 256)]);
            current = code(&shifted, 75).decoded;
            offset = (offset + step) % 256;
            // undo the accumulated shift to compare with the original
            let aligned = Image::from_fn(256, 256, |x, y| current[((x + 256 - offset) % 256, (y + 256 - offset) % 256)]);
            results.push(psnr(&img, &aligned, 255.0));
        }
        println!("10 generations at quality 75, {name}: PSNR after 1, 2, 5, 10: {:.1}, {:.1}, {:.1}, {:.1} dB", results[0], results[1], results[4], results[9]);
    }
    println!("wrote pictures to {dir}/");
}
