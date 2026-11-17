// Model solution for the coding exercises of "Fast MRI with compressed sensing".
// k-space is the 2D DFT of the image (`imgcourse::fft::fft2`); the scanner measures whole
// rows of it ("lines", one row index `ky` each). A mask says which rows were measured.

use imgcourse::fft::{fft2, fft2_real, frequency};
use imgcourse::{Complex, Image, Rng};

/// C1. A random variable-density mask of `n` k-space lines: all lines with `|frequency| ≤
/// centre/2`, then further lines drawn at random with probability ∝ `(1 − |f|/(n/2))^q`
/// until at least `lines` are measured. Line `k` and its partner `n − k` (frequency −f)
/// are always taken together.
pub fn variable_density_mask(n: usize, lines: usize, centre: usize, q: f64, seed: u64) -> Vec<bool> {
    let mut mask: Vec<bool> = (0..n).map(|k| frequency(k, n).abs() <= centre as f32 / 2.0).collect();
    let mut rng = Rng::new(seed);
    let weight = |k: usize| (1.0 - frequency(k, n).abs() as f64 / (n as f64 / 2.0)).powf(q);
    while mask.iter().filter(|m| **m).count() < lines {
        // Draw one of the lines not yet measured (positive frequencies 1..=n/2), each with its weight.
        let total: f64 = (1..=n / 2).filter(|&k| !mask[k]).map(weight).sum();
        if total <= 0.0 {
            break; // nothing left to draw
        }
        let mut r = rng.uniform() as f64 * total;
        for k in 1..=n / 2 {
            if mask[k] {
                continue;
            }
            r -= weight(k);
            if r <= 0.0 {
                mask[k] = true;
                mask[n - k] = true;
                break;
            }
        }
    }
    mask
}

/// C2. The zero-filled image: the 2D DFT of `img`, the rows `ky` with `mask[ky] == false`
/// set to 0, inverse DFT, real part.
pub fn zero_filled(img: &Image<f32>, mask: &[bool]) -> Image<f32> {
    let mut k = fft2_real(img);
    for (kx, ky) in k.coords() {
        if !mask[ky] {
            k[(kx, ky)] = Complex::ZERO;
        }
    }
    fft2(&k, true).map(|c| c.re)
}

/// C3. Data consistency, the gradient step of ISTA/FISTA with step 1: the 2D DFT of `x`, its
/// measured rows replaced by the data `kspace`, inverse DFT, real part.
pub fn data_consistency(x: &Image<f32>, kspace: &Image<Complex>, mask: &[bool]) -> Image<f32> {
    let mut k = fft2_real(x);
    for (kx, ky) in k.coords() {
        if mask[ky] {
            k[(kx, ky)] = kspace[(kx, ky)];
        }
    }
    fft2(&k, true).map(|c| c.re)
}

/// C4. The compressed-sensing reconstruction by FISTA: start from the zero-filled image;
/// each iteration takes a data-consistency step from the extrapolated image `z`, then
/// soft-thresholds the Haar detail coefficients (`levels` levels) at λ and transforms back;
/// then the momentum step as in the solvers unit.
pub fn cs_reconstruct(kspace: &Image<Complex>, mask: &[bool], lambda: f32, levels: usize, iterations: usize) -> Image<f32> {
    let mut x = fft2(kspace, true).map(|c| c.re); // zero filled (the data are 0 off the mask)
    let mut z = x.clone();
    let mut t = 1.0f32;
    for _ in 0..iterations {
        let g = data_consistency(&z, kspace, mask);
        let c = haar2d(&g, levels);
        let coarse = g.width() >> levels; // the top-left block holds the averages: keep it
        let shrunk = c.map_xy(|x, y, v| {
            if x < coarse && y < coarse {
                v
            } else {
                v.signum() * (v.abs() - lambda).max(0.0)
            }
        });
        let x_new = inverse_haar2d(&shrunk, levels);
        let t_new = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let m = (t - 1.0) / t_new;
        z = x_new.zip_map(&x, |a, b| a + m * (a - b));
        x = x_new;
        t = t_new;
    }
    x
}

/// The orthonormal 2D Haar transform (as on the wavelets day), `levels` levels, for a square
/// image whose size is divisible by 2^levels. Each level splits the top-left block into
/// averages (top left) and three blocks of details.
pub fn haar2d(img: &Image<f32>, levels: usize) -> Image<f32> {
    let mut c = img.clone();
    let mut size = img.width();
    for _ in 0..levels {
        haar_level(&mut c, size, false);
        size /= 2;
    }
    c
}

/// The inverse of `haar2d`.
pub fn inverse_haar2d(c: &Image<f32>, levels: usize) -> Image<f32> {
    let mut img = c.clone();
    for l in (0..levels).rev() {
        haar_level(&mut img, c.width() >> l, true);
    }
    img
}

/// One level of the 2D Haar transform (or its inverse) on the top-left `size × size` block:
/// along every row, then along every column.
fn haar_level(c: &mut Image<f32>, size: usize, inverse: bool) {
    let h = size / 2;
    let r = std::f32::consts::FRAC_1_SQRT_2;
    for pass in 0..2 {
        let along_rows = (pass == 0) != inverse; // the inverse undoes the columns first
        for line in 0..size {
            let get = |c: &Image<f32>, i: usize| if along_rows { c[(i, line)] } else { c[(line, i)] };
            let old: Vec<f32> = (0..size).map(|i| get(c, i)).collect();
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
