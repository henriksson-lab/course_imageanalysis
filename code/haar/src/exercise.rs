// Coding exercises for "Localised basis functions and the Haar transform". Replace each
// `todo!()` with your code, then check it with `cargo test -p haar` and look at `cargo run -p haar`.
#![allow(unused_variables, unused_imports)]

/// The scale of one Haar step: `1/2` by hand (averages and half-differences), `1/√2` for the
/// orthonormal transform.
pub fn scale(orthonormal: bool) -> f32 {
    if orthonormal {
        std::f32::consts::FRAC_1_SQRT_2
    } else {
        0.5
    }
}

/// C1a. One Haar step: every pair `(a, b)` = `(x[2k], x[2k+1])` becomes the average
/// `s = (a + b)·scale` and the difference `d = (a − b)·scale` (slide "The Haar transform by hand").
/// Returns `(s, d)`, each half as long as `x`.
pub fn haar_step(x: &[f32], orthonormal: bool) -> (Vec<f32>, Vec<f32>) {
    todo!("C1a: haar_step")
}

/// C1b. The full Haar transform of `N = 2^J` samples: repeat the step on the averages until one
/// is left. The result is `(s | d_J | ... | d_2 | d_1)`, coarse to fine: the overall average
/// first, the finest differences in the second half.
///
/// Hint: work in one vector `c`, starting as a copy of `x`, and a length `len = N`. While
/// `len > 1`: step on `c[..len]`, write `s` to `c[..len/2]` and `d` to `c[len/2..len]`, halve `len`.
/// `c[a..b].copy_from_slice(&v)` copies a vector into a piece of `c`.
pub fn haar_forward(x: &[f32], orthonormal: bool) -> Vec<f32> {
    todo!("C1b: haar_forward")
}

/// C2. The inverse transform: from the coarsest level up, every `(s, d)` becomes the pair
/// `a = s + d`, `b = s − d` (by hand), or `(s + d)/√2`, `(s − d)/√2` (orthonormal)
/// (slide "Going back: nothing is lost").
///
/// Hint: the reverse of C1b. Start with `len = 1`; while `len < N`, the averages are
/// `x[..len]` and the differences `x[len..2·len]`; replace `x[..2·len]` by the pairs, double `len`.
pub fn haar_inverse(c: &[f32], orthonormal: bool) -> Vec<f32> {
    todo!("C2: haar_inverse")
}

/// C3. Keep the `k` coefficients of largest magnitude and set the others to 0. Returns the kept
/// vector and the sum of the squares of the dropped coefficients (in an orthonormal basis: the
/// squared error of the reconstruction, slide "The math: error of keeping K coefficients").
///
/// Hint: sort the positions `0..n` by magnitude, largest first:
/// `order.sort_by(|&i, &j| c[j].abs().total_cmp(&c[i].abs()))`.
pub fn keep_largest(c: &[f32], k: usize) -> (Vec<f32>, f32) {
    todo!("C3: keep_largest")
}

/// C4. The orthonormal Haar basis vectors of length `n`, as the rows of a matrix: row `i` is the
/// signal whose coefficients are 1 at position `i` and 0 elsewhere (slide "The Haar basis vectors").
///
/// Hint: one call of `haar_inverse` per row.
pub fn haar_basis(n: usize) -> Vec<Vec<f32>> {
    todo!("C4: haar_basis")
}
