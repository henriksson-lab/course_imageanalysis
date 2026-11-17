// The reversible 5/3 wavelet on the micrograph: exact reconstruction, the smaller images
// inside one code stream, and the entropy of the coefficients. Also bit planes and chunks.
// Pictures go to out/compress-science/.
//   cargo run -p day08 -- compress-science                      (your code)
//   cargo run -p day08 --features solution -- compress-science  (the model solution)

use crate::compress_science::*;
use imgcourse::*;

/// The entropy in bits per value of a list of whole numbers.
fn entropy(values: &[i32]) -> f64 {
    let lo = *values.iter().min().unwrap();
    let hi = *values.iter().max().unwrap();
    let mut counts = vec![0usize; (hi - lo + 1) as usize];
    for &v in values {
        counts[(v - lo) as usize] += 1;
    }
    let n = values.len() as f64;
    counts.iter().filter(|&&c| c > 0).map(|&c| -(c as f64 / n) * (c as f64 / n).log2()).sum()
}

pub fn main() {
    let dir = "out/compress-science";
    let img = samples::cell().map(|v| (v * 255.0).round() as i32);

    // 1. Exactly lossless: transform every row and back.
    let exact = (0..img.height()).all(|y| {
        let (s, d) = lift53(img.row(y));
        unlift53(&s, &d) == img.row(y)
    });
    println!("every row of the micrograph comes back exactly: {exact}");

    // 2. Three levels: each time the wavelet of the LL quarter. The LL quarters are the
    // smaller images that a decoder can read without the rest of the stream.
    let mut all = img.clone();
    let mut size = img.width();
    for level in 1..=3 {
        let ll = all.crop(0, 0, size, size);
        let w = wavelet2d(&ll);
        for (x, y) in w.coords() {
            all[(x, y)] = w[(x, y)];
        }
        size /= 2;
        let small = all.crop(0, 0, size, size);
        save_gray(&small.map(|v| v as f32 / 255.0), format!("{dir}/ll-level{level}-{size}.png"));
    }
    // all coefficients: the LL band as it is, the details × 2 around mid grey
    let picture = all.map_xy(|x, y, v| if x < size && y < size { v as f32 / 255.0 } else { 0.5 + 2.0 * v as f32 / 255.0 });
    save_gray(&picture, format!("{dir}/wavelet-3-levels.png"));
    println!("entropy of the pixels:                        {:.2} bits/pixel", entropy(img.pixels()));
    println!("entropy of the 3-level wavelet coefficients:  {:.2} bits/pixel", entropy(all.pixels()));

    // 3. Bit planes: the guess for a coefficient of 45 (6 bits) as the planes arrive.
    print!("bit planes, value 45 of 6 bits:");
    for planes in 1..=6 {
        print!("  {planes}: {}", bitplane_guess(45, 6, planes));
    }
    println!();

    // 4. Chunks: reading from a 2048 × 2048 × 100 stack.
    let reads = [("512 × 512 region of one slice", [(0, 512), (0, 512), (40, 41)]), ("x–z plane (fixed y)", [(0, 2048), (700, 701), (0, 100)])];
    for (name, region) in reads {
        let needed: usize = region.iter().map(|(a, b)| b - a).product();
        for chunk in [[256, 256, 1], [64, 64, 64]] {
            let n = chunks_read(&region, &chunk);
            let decoded = n * chunk.iter().product::<usize>();
            println!("{name}, chunks {chunk:?}: {n} chunks, {:.2} MB decoded for {:.2} MB needed", decoded as f64 * 2e-6, needed as f64 * 2e-6);
        }
    }
    println!("wrote pictures to {dir}/");
}
