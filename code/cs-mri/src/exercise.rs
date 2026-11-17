// Coding exercises for "Fast MRI with compressed sensing". Replace each `todo!()` with your
// code, then check it with `cargo test -p cs-mri` and look at `cargo run -p cs-mri`.
// k-space is the 2D DFT of the image (`imgcourse::fft::fft2`); the scanner measures whole
// rows of it ("lines", one row index `ky` each). A mask says which rows were measured.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::{fft2, fft2_real, frequency};
use imgcourse::{Complex, Image, Rng};

/// C1. A random variable-density mask of `n` k-space lines: all lines with `|frequency| ≤
/// centre/2`, then further lines drawn at random with probability ∝ `(1 − |f|/(n/2))^q`
/// until at least `lines` are measured. Line `k` and its partner `n − k` (frequency −f)
/// are always taken together.
///
/// Hint: `frequency(k, n)` is the frequency of row `k` (negative for `k > n/2`). To draw one
/// line with probability ∝ weight: add up the weights of the lines `k = 1 ..= n/2` not yet
/// taken, pick `r = rng.uniform() × total`, and walk through those lines subtracting their
/// weights until `r ≤ 0`. Mark `k` and `n − k`.
pub fn variable_density_mask(n: usize, lines: usize, centre: usize, q: f64, seed: u64) -> Vec<bool> {
    todo!("C1: variable_density_mask")
}

/// C2. The zero-filled image: the 2D DFT of `img`, the rows `ky` with `mask[ky] == false`
/// set to 0, inverse DFT, real part.
///
/// Hint: `fft2_real(img)` gives k-space as an `Image<Complex>`, with `ky` as the row: the
/// entry `(kx, ky)`. `fft2(&k, true)` transforms back; `.map(|c| c.re)` keeps the real part.
pub fn zero_filled(img: &Image<f32>, mask: &[bool]) -> Image<f32> {
    todo!("C2: zero_filled")
}

/// C3. Data consistency, the gradient step of ISTA/FISTA with step 1: the 2D DFT of `x`, its
/// measured rows replaced by the data `kspace`, inverse DFT, real part.
pub fn data_consistency(x: &Image<f32>, kspace: &Image<Complex>, mask: &[bool]) -> Image<f32> {
    todo!("C3: data_consistency")
}

/// C4. The compressed-sensing reconstruction by FISTA: start from the zero-filled image;
/// each iteration takes a data-consistency step from the extrapolated image `z`, then
/// soft-thresholds the Haar detail coefficients (`levels` levels) at λ and transforms back;
/// then the momentum step as in the solvers unit.
///
/// Hint: `haar2d` and `inverse_haar2d` are below. Keep the top-left block of size
/// `width >> levels` (the averages) unchanged. The momentum step: `t_new = (1 + √(1 + 4t²))/2`,
/// `z = x_new + (t − 1)/t_new · (x_new − x)` (use `zip_map`).
pub fn cs_reconstruct(kspace: &Image<Complex>, mask: &[bool], lambda: f32, levels: usize, iterations: usize) -> Image<f32> {
    todo!("C4: cs_reconstruct")
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
