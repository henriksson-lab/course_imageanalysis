// Shrinks the 2D Haar coefficients of the micrograph with the three penalties; prints how
// sparse the micrograph is and writes the denoised images to out/sparsity/.
//   cargo run -p sparse -- sparsity                      (your code)
//   cargo run -p sparse --features solution -- sparsity  (the model solution)

use imgcourse::*;
use crate::sparsity::*;

pub fn main() {
    let dir = "out/sparsity";
    let img = samples::cell();
    let n = img.width();
    let levels = 4;

    // The by-hand exercises.
    let w = [4.0, -0.5, 1.2, -2.5, 0.3, 3.0];
    println!("A1, λ = 1:");
    println!("  ℓ2: {:.2?}", w.map(|v| shrink_l2(v, 1.0)));
    println!("  ℓ1: {:.2?}", w.map(|v| soft(v, 1.0)));
    println!("  ℓ0: {:.2?}", w.map(|v| hard(v, 1.0)));
    println!("A3, closest point of the ℓ1 ball to (3, 1):");
    for budget in [1.0, 2.0, 3.0, 4.0] {
        println!("  budget {budget}: {:.2?}", project_l1_ball(&[3.0, 1.0], budget));
    }

    // How sparse is the micrograph in the 2D Haar basis? (Mean removed, so that the one
    // large coarse coefficient does not count.)
    let mean = img.mean() as f64;
    let c = haar2d(&img.map(|v| v as f64 - mean), levels);
    println!("micrograph, {} Haar coefficients (mean removed):", n * n);
    for f in [0.9, 0.99, 0.999] {
        let k = count_for_energy(c.pixels(), f);
        println!("  {:.1} % of the energy in the largest {k} ({:.1} %)", 100.0 * f, 100.0 * k as f64 / (n * n) as f64);
    }

    // Denoising: shrink the detail coefficients with each penalty; keep the coarsest block.
    // For each penalty, try a range of λ and keep the best (by the error against the truth).
    let sigma = 20.0 / 255.0;
    let noisy = samples::add_noise(&img, sigma, 1);
    save_gray(&img, format!("{dir}/input.png"));
    save_gray(&noisy, format!("{dir}/noisy.png"));
    println!("noise σ = 20 grey levels; noisy image: PSNR {:.2} dB", psnr(&noisy, &img));
    let cn = haar2d(&noisy.map(|v| v as f64), levels);
    let coarse = n >> levels; // the top-left coarse × coarse block is not shrunk
    let penalties: [(&str, Shrink); 3] = [("l2", shrink_l2), ("l1", soft), ("l0", hard)];
    for (name, f) in penalties {
        let mut best = (f64::NEG_INFINITY, 0.0, 0, Image::new(n, n, 0.0f32));
        for step in 1..=40 {
            // ℓ2 needs a much wider range of λ: (λ/2)c² is a different scale from λ|c|.
            let lambda = if name == "l2" { 0.05 * step as f64 } else { 0.005 * step as f64 };
            let shrunk = c_map(&cn, coarse, |v| f(v, lambda));
            let zeros = shrunk.pixels().iter().filter(|&&v| v == 0.0).count();
            let out = inverse_haar2d(&shrunk, levels).map(|v| v as f32);
            let p = psnr(&out, &img);
            if p > best.0 {
                best = (p, lambda, zeros, out);
            }
        }
        let (p, lambda, zeros, out) = best;
        println!(
            "  {name}: best λ = {lambda:.3} ({:.1} grey levels): PSNR {p:.2} dB, {:.1} % of the coefficients exactly 0",
            lambda * 255.0,
            100.0 * zeros as f64 / (n * n) as f64
        );
        save_gray(&out, format!("{dir}/denoised-{name}.png"));
        // A close-up, enlarged 3 times, to see the texture and the noise.
        let zoom = |im: &Image<f32>| Image::from_fn(240, 240, |x, y| im[(60 + x / 3, 90 + y / 3)]);
        save_gray(&zoom(&out), format!("{dir}/zoom-{name}.png"));
        if name == "l0" {
            save_gray(&zoom(&noisy), format!("{dir}/zoom-noisy.png"));
            save_gray(&zoom(&img), format!("{dir}/zoom-input.png"));
        }
    }
    println!("wrote the results to {dir}/");
}

/// A rule that shrinks one coefficient `w` with the weight λ.
type Shrink = fn(f64, f64) -> f64;

/// Apply `f` to every coefficient except the coarsest `coarse × coarse` block.
fn c_map(c: &Image<f64>, coarse: usize, f: impl Fn(f64) -> f64) -> Image<f64> {
    c.map_xy(|x, y, v| if x < coarse && y < coarse { v } else { f(v) })
}

/// Peak signal-to-noise ratio in dB (peak 1).
fn psnr(a: &Image<f32>, truth: &Image<f32>) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.coords() {
        sum += ((a[(x, y)] - truth[(x, y)]) as f64).powi(2);
    }
    let mse = sum / (a.width() * a.height()) as f64;
    10.0 * (1.0 / mse).log10()
}

/// The orthonormal 2D Haar transform (as on the wavelets day), `levels` levels, for a square
/// image whose size is divisible by 2^levels. Each level splits the top-left block into
/// averages (top left) and three blocks of details.
fn haar2d(img: &Image<f64>, levels: usize) -> Image<f64> {
    let mut c = img.clone();
    let mut size = img.width();
    for _ in 0..levels {
        haar_level(&mut c, size, false);
        size /= 2;
    }
    c
}

/// The inverse of `haar2d`.
fn inverse_haar2d(c: &Image<f64>, levels: usize) -> Image<f64> {
    let mut img = c.clone();
    for l in (0..levels).rev() {
        haar_level(&mut img, c.width() >> l, true);
    }
    img
}

/// One level of the 2D Haar transform (or its inverse) on the top-left `size × size` block:
/// along every row, then along every column.
fn haar_level(c: &mut Image<f64>, size: usize, inverse: bool) {
    let h = size / 2;
    let r = std::f64::consts::FRAC_1_SQRT_2;
    for pass in 0..2 {
        let along_rows = (pass == 0) != inverse; // the inverse undoes the columns first
        for line in 0..size {
            let get = |c: &Image<f64>, i: usize| if along_rows { c[(i, line)] } else { c[(line, i)] };
            let old: Vec<f64> = (0..size).map(|i| get(c, i)).collect();
            let mut new = vec![0.0; size];
            for i in 0..h {
                if inverse {
                    new[2 * i] = r * (old[i] + old[h + i]);
                    new[2 * i + 1] = r * (old[i] - old[h + i]);
                } else {
                    new[i] = r * (old[2 * i] + old[2 * i + 1]);
                    new[h + i] = r * (old[2 * i] - old[2 * i + 1]);
                }
            }
            for (i, v) in new.into_iter().enumerate() {
                if along_rows {
                    c[(i, line)] = v;
                } else {
                    c[(line, i)] = v;
                }
            }
        }
    }
}
