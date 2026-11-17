// Coding exercises for "Sparse coding and dictionary learning". Replace each `todo!()` with
// your code, then check it with `cargo test -p day11 dictionaries::` and look at `cargo run -p day11 -- dictionaries`.
// A patch is a vector (`Vec<f64>`, the pixels row by row); a dictionary is a list of atoms,
// each a unit vector of the same length.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Image};

/// The dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The reconstruction `Σ c_j d_j` of a code.
pub fn rebuild(dict: &[Vec<f64>], code: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dict[0].len()];
    for (atom, &c) in dict.iter().zip(code) {
        if c != 0.0 {
            for (o, a) in out.iter_mut().zip(atom) {
                *o += c * a;
            }
        }
    }
    out
}

/// C1. Sparse coding by orthogonal matching pursuit: pick the atom with the largest
/// `|d_jᵀ r|`, refit all chosen atoms by least squares, update the residual `r`; stop after
/// `max_atoms` atoms or when `‖r‖² ≤ max_error`. Returns one coefficient per atom.
///
/// Hint: the refit is `linalg::least_squares(&a, y)` where `a` has one row per pixel and one
/// column per chosen atom: `a[i][k] = dict[chosen[k]][i]`. It returns an `Option`. `rebuild`
/// and `dot` above help with the residual.
pub fn omp(dict: &[Vec<f64>], y: &[f64], max_atoms: usize, max_error: f64) -> Vec<f64> {
    todo!("C1: omp")
}

/// C2. The K-SVD update of one atom: the unit vector `d` that best fits the `residuals` of
/// the patches that use it, with free coefficients. `d` is the top eigenvector of
/// `Σ r rᵀ` (found by power iteration, starting from `start`); the coefficients are the
/// projections `rᵀd`. Returns `(d, coefficients)`.
///
/// Hint: power iteration as in the PCA unit, 50 steps: `v = Σ r (rᵀd)` (this is `(Σ r rᵀ) d`
/// without building the matrix), then `d = v / ‖v‖`.
pub fn update_atom(residuals: &[Vec<f64>], start: &[f64]) -> (Vec<f64>, Vec<f64>) {
    todo!("C2: update_atom")
}

/// C3. K-SVD: `iterations` times, code every patch with OMP (`k` atoms each), then update
/// every atom in turn: collect the patches that use it, subtract the contributions of their
/// other atoms, and replace the atom and its coefficients by `update_atom`. Returns the
/// dictionary and the mean squared error per pixel after each iteration.
///
/// Hint: the codes come from `omp(&dict, p, k, 0.0)`. For atom `j`, the users are the patches
/// with `codes[i][j] != 0`; a user's residual is `patch − rebuild(&dict, &codes[i]) +
/// codes[i][j] · dict[j]`. After `update_atom`, write the new coefficients into `codes[i][j]`.
/// The error: `Σ ‖patch − rebuild‖²` over all patches, divided by (patches × pixels).
pub fn ksvd(patches: &[Vec<f64>], start: &[Vec<f64>], k: usize, iterations: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    todo!("C3: ksvd")
}

/// C4. Denoise by coding every overlapping patch (step 1) with OMP until the residual is
/// at the noise level, `d (1.15 σ)²` for patches of `d` pixels (at most `d` atoms), then
/// average: every pixel becomes `(λ y + Σ of the patch reconstructions that cover it) /
/// (λ + the number of patches that cover it)`. The patch size comes from the atoms (`d` =
/// size²).
///
/// Hint: `img.crop(x0, y0, size, size).pixels()` gives a patch row by row; pixel `i` of the
/// patch is at `(x0 + i % size, y0 + i / size)`. Keep two images, the sum and the count.
pub fn denoise(img: &Image<f32>, dict: &[Vec<f64>], sigma: f64, lambda: f64) -> Image<f32> {
    todo!("C4: denoise")
}
