// Learns a dictionary of 8 × 8 atoms from the micrograph with K-SVD, draws the atoms, and
// denoises a noisy crop with the learned and with a fixed (DCT) dictionary. Results in
// out/dictionaries/.
//   cargo run -p day11 -- dictionaries                      (your code)
//   cargo run -p day11 --features solution -- dictionaries  (the model solution)

use crate::dictionaries::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/dictionaries";
    let img = samples::cell();
    let size = 8;
    let d = size * size;

    // Training patches: every 2nd position in the 128 × 128 centre of the micrograph.
    let train = img.crop(64, 64, 128, 128);
    let mut patches = Vec::new();
    for y0 in (0..=128 - size).step_by(2) {
        for x0 in (0..=128 - size).step_by(2) {
            patches.push(train.crop(x0, y0, size, size).pixels().iter().map(|&v| v as f64).collect::<Vec<f64>>());
        }
    }

    // Start: the constant atom and 63 random training patches, scaled to length 1.
    let mut rng = Rng::new(1);
    let mut start = vec![vec![1.0 / size as f64; d]];
    while start.len() < 64 {
        let p = &patches[rng.below(patches.len())];
        let len = dot(p, p).sqrt();
        start.push(p.iter().map(|v| v / len).collect());
    }
    let k = 4;
    let (dict, errors) = ksvd(&patches, &start, k, 8);
    let gl = |e: f64| e * 255.0 * 255.0; // squared error in grey levels²
    println!("{} patches, 64 atoms, K = {k}: error per pixel (grey levels²) by iteration:", patches.len());
    println!("  {}", errors.iter().map(|e| format!("{:.0}", gl(*e))).collect::<Vec<_>>().join(", "));
    save_rgb(&atom_sheet(&start), format!("{dir}/atoms-start.png"));
    save_rgb(&atom_sheet(&dict), format!("{dir}/atoms-learned.png"));

    // The same error with the fixed 64 DCT atoms (the JPEG basis) and K = 4.
    let dct = dct_atoms(size);
    let err_dct: f64 = patches.iter().map(|p| {
        let r = rebuild(&dct, &omp(&dct, p, k, 0.0));
        p.iter().zip(r).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
    }).sum::<f64>() / (patches.len() * d) as f64;
    println!("  the 64 DCT atoms, best {k} per patch: {:.0}", gl(err_dct));

    // Denoise another region (96 × 96 at the bottom left), noise σ = 20 grey levels.
    let sigma = 20.0 / 255.0;
    let clean = img.crop(0, 160, 96, 96);
    let noisy = samples::add_noise(&clean, sigma as f32, 2);
    save_gray(&clean, format!("{dir}/clean.png"));
    save_gray(&noisy, format!("{dir}/noisy.png"));
    println!("denoising a 96 × 96 crop, σ = 20: noisy {:.2} dB", psnr(&noisy, &clean));
    for (name, dictionary) in [("dct", &dct), ("learned", &dict)] {
        let out = denoise(&noisy, dictionary, sigma, 0.5);
        println!("  {name:7} dictionary: {:.2} dB", psnr(&out, &clean));
        save_gray(&out, format!("{dir}/denoised-{name}.png"));
    }
    println!("wrote the results to {dir}/");
}

/// Peak signal-to-noise ratio in dB (peak 1).
fn psnr(a: &Image<f32>, truth: &Image<f32>) -> f64 {
    let mse = a.zip_map(truth, |p, q| (p - q) * (p - q)).mean() as f64;
    10.0 * (1.0 / mse).log10()
}

/// The 2D DCT basis for `size × size` patches, as atoms (unit vectors), lowest frequencies first.
fn dct_atoms(size: usize) -> Vec<Vec<f64>> {
    let n = size as f64;
    let c = |k: usize, i: usize| {
        let s = if k == 0 { (1.0 / n).sqrt() } else { (2.0 / n).sqrt() };
        s * (std::f64::consts::PI * (2 * i + 1) as f64 * k as f64 / (2.0 * n)).cos()
    };
    let mut atoms = Vec::new();
    for ky in 0..size {
        for kx in 0..size {
            atoms.push((0..size * size).map(|p| c(kx, p % size) * c(ky, p / size)).collect());
        }
    }
    atoms
}

/// The atoms side by side, 8 per row, each enlarged 4 times and stretched from its smallest
/// value (black) to its largest (white).
fn atom_sheet(atoms: &[Vec<f64>]) -> Image<[f32; 3]> {
    let size = (atoms[0].len() as f64).sqrt() as usize;
    let (cell, zoom) = (size * 4 + 2, 4);
    let rows = atoms.len().div_ceil(8);
    let mut sheet = Image::new(8 * cell, rows * cell, [1.0f32; 3]);
    for (a, atom) in atoms.iter().enumerate() {
        let lo = atom.iter().fold(f64::INFINITY, |m, v| m.min(*v));
        let hi = atom.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        for (i, v) in atom.iter().enumerate() {
            for dy in 0..zoom {
                for dx in 0..zoom {
                    let x = (a % 8) * cell + 1 + (i % size) * zoom + dx;
                    let y = (a / 8) * cell + 1 + (i / size) * zoom + dy;
                    let g = if hi - lo > 1e-9 { (v - lo) / (hi - lo) } else { 0.5 }; // a constant atom: grey
                    sheet[(x, y)] = [g as f32; 3];
                }
            }
        }
    }
    sheet
}
