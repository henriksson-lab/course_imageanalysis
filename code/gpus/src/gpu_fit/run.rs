// Irregular algorithms in parallel passes: labelling a spiral and blobs, the DP wavefront,
// Amdahl's law, and float16 sums. Numbers on the terminal, pictures in out/gpu-fit/.
//   cargo run -p gpus -- gpu-fit                      (your code)
//   cargo run -p gpus --features solution -- gpu-fit  (the model solution)

use crate::gpu_fit::*;
use imgcourse::*;

/// A one-pixel-wide square spiral in an `n` × `n` image, with one-pixel gaps.
fn spiral(n: usize) -> Image<bool> {
    let mut m = Image::new(n, n, false);
    let (mut x, mut y) = (0isize, 0isize);
    let dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    let mut d = 0;
    m[(0, 0)] = true;
    let free = |m: &Image<bool>, x: isize, y: isize| m.contains(x, y) && !m[(x as usize, y as usize)];
    loop {
        let mut moved = false;
        for _ in 0..2 {
            let (dx, dy) = dirs[d];
            let (nx, ny) = (x + dx, y + dy);
            // Go on if the next pixel is free and the one after it is not part of the spiral.
            if free(&m, nx, ny) && free(&m, nx + dx, ny + dy) || (free(&m, nx, ny) && !m.contains(nx + dx, ny + dy)) {
                x = nx;
                y = ny;
                m[(x as usize, y as usize)] = true;
                moved = true;
                break;
            }
            d = (d + 1) % 4; // turn right
        }
        if !moved {
            return m;
        }
    }
}

pub fn main() {
    let dir = "out/gpu-fit";
    let s = spiral(24);
    let (_, labels) = samples::nuclei(256, 30, 5);
    let blobs = labels.map(|l| l > 0);
    for (name, mask) in [("spiral 24 × 24", &s), ("nuclei 256 × 256", &blobs)] {
        let pixels = mask.pixels().iter().filter(|&&b| b).count();
        let (l, plain) = label_propagation(mask);
        let (_, jumping) = label_pointer_jumping(mask);
        let mut ids: Vec<u32> = l.pixels().iter().copied().filter(|&v| v > 0).collect();
        ids.sort();
        ids.dedup();
        println!("{name}: {pixels} foreground pixels, {} components; propagation {plain} passes, with pointer jumping {jumping}", ids.len());
    }
    save_gray(&s.map(|b| if b { 1.0 } else { 0.0 }), format!("{dir}/spiral.png"));
    // The pass in which each pixel of the spiral got its final label.
    let (w, h) = (s.width(), s.height());
    let mut labels = Image::from_fn(w, h, |x, y| if s[(x, y)] { (y * w + x + 1) as u32 } else { 0 });
    let mut settled = Image::new(w, h, 0.0f32);
    let mut pass = 0;
    loop {
        let (next, changed) = propagate_once(&s, &labels);
        if !changed {
            break;
        }
        pass += 1;
        for (x, y) in s.coords() {
            if next[(x, y)] != labels[(x, y)] {
                settled[(x, y)] = pass as f32;
            }
        }
        labels = next;
    }
    let big = Image::from_fn(10 * w, 10 * h, |x, y| settled[(x / 10, y / 10)]);
    save_gray_stretched(&big, format!("{dir}/spiral-passes.png"));

    for (rows, cols) in [(10, 16), (1000, 1000)] {
        let (_, cells) = dp_wavefront(&Image::new(cols, rows, 1.0f32));
        println!(
            "DP table {rows} × {cols}: {} steps, at most {} cells per step, {:.1} on average",
            cells.len(),
            cells.iter().max().unwrap(),
            (rows * cols) as f32 / cells.len() as f32
        );
    }

    println!("Amdahl: speed-up for a serial fraction s and the rest k times faster");
    for serial in [0.4, 0.2, 0.1, 0.01] {
        println!("  s = {serial:4}: k = 10: {:5.2}, k = 50: {:5.2}, k = 1000: {:6.2}, limit {:5.1}", amdahl(serial, 10.0), amdahl(serial, 50.0), amdahl(serial, 1000.0), 1.0 / serial);
    }

    println!("Summing 0.01 again and again:");
    for n in [1000, 3000, 10000, 100000] {
        let v = vec![0.01f32; n];
        println!("  {n:6} terms: float16 {:8.3}, float16 values with a float32 sum {:9.3}, exact {:7.2}", sum_f16(&v), sum_mixed(&v), 0.01 * n as f64);
    }
    println!("wrote the pictures to {dir}/");
}
