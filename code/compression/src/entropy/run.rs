// Entropy and entropy codes on real images: the micrograph, the same pixels shuffled, and a
// binary mask. Numbers are printed; pictures go to out/entropy/.
//   cargo run -p compression -- entropy                      (your code)
//   cargo run -p compression --features solution -- entropy  (the model solution)

use crate::entropy::*;
use imgcourse::*;

/// The average Huffman code length in bits per symbol.
fn huffman_mean(counts: &[usize]) -> f32 {
    let lengths = huffman_lengths(counts);
    let total: usize = counts.iter().sum();
    counts.iter().zip(&lengths).map(|(&c, &l)| (c * l) as f32).sum::<f32>() / total as f32
}

pub fn main() {
    let dir = "out/entropy";

    // 1. The micrograph: entropy and Huffman code of its grey levels.
    let img = samples::cell();
    save_gray(&img, format!("{dir}/cell.png"));
    let counts = histogram(&img);
    println!("micrograph: {} grey levels used", counts.iter().filter(|&&c| c > 0).count());
    println!("  entropy H = {:.3} bits/pixel, Huffman {:.3} bits/pixel, raw 8", entropy(&counts), huffman_mean(&counts));

    // 2. The same pixels in a random order: the same histogram, so the same entropy.
    let mut pixels = img.pixels().to_vec();
    let mut rng = Rng::new(7);
    for i in (1..pixels.len()).rev() {
        pixels.swap(i, rng.below(i + 1));
    }
    let shuffled = Image::from_vec(img.width(), img.height(), pixels);
    save_gray(&shuffled, format!("{dir}/cell-shuffled.png"));
    println!("shuffled:   entropy H = {:.3} bits/pixel", entropy(&histogram(&shuffled)));

    // 3. A binary mask (the nuclei of a synthetic fluorescence image).
    let (_, labels) = samples::nuclei(256, 6, 3);
    let mask: Image<u8> = labels.map(|l| u8::from(l > 0));
    save_gray(&mask.map(|v| v as f32), format!("{dir}/mask.png"));
    let n = (mask.width() * mask.height()) as f32;
    let ones = mask.pixels().iter().filter(|&&v| v == 1).count();
    let h = entropy(&[mask.pixels().len() - ones, ones]);
    println!("mask: {:.1} % object pixels", 100.0 * ones as f32 / n);
    println!("  entropy H = {h:.3} bits/pixel; Huffman on single pixels: {:.3} bits/pixel", huffman_mean(&[mask.pixels().len() - ones, ones]));

    // Run lengths of every row, then a Huffman code for the run lengths.
    let mut runs = Vec::new();
    for y in 0..mask.height() {
        runs.extend(run_lengths(mask.row(y)));
    }
    let longest = *runs.iter().max().unwrap();
    let mut run_counts = vec![0; longest + 1];
    for &r in &runs {
        run_counts[r] += 1;
    }
    let bits = huffman_mean(&run_counts) * runs.len() as f32;
    println!("  {} runs; Huffman-coded run lengths: {:.3} bits/pixel", runs.len(), bits / n);

    // Blocks of 2 × 2 pixels as one symbol (16 possible symbols).
    let mut block_counts = vec![0; 16];
    for y in (0..mask.height()).step_by(2) {
        for x in (0..mask.width()).step_by(2) {
            let s = mask[(x, y)] + 2 * mask[(x + 1, y)] + 4 * mask[(x, y + 1)] + 8 * mask[(x + 1, y + 1)];
            block_counts[s as usize] += 1;
        }
    }
    println!("  Huffman on 2 × 2 blocks: {:.3} bits/pixel", huffman_mean(&block_counts) / 4.0);

    println!("wrote pictures to {dir}/");
}
