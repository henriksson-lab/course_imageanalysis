// Coding exercises for "From a peak to a mosaic". Replace each `todo!()` with your code,
// then check it with `cargo test -p fftreg stitching::` and look at `cargo run -p fftreg -- stitching`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::fft2;
use imgcourse::{linalg, Complex, Image};

/// Phase correlation of two tiles of the same size (given, from the last unit): means
/// subtracted, no taper, the cross-power normalised to size 1. `out[(p, q)]` is the score of the
/// shift `(p, q)` modulo the tile size.
pub fn phase_correlation(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    let (ma, mb) = (a.mean(), b.mean());
    let fa = fft2(&a.map(|v| Complex::from(v - ma)), false);
    let fb = fft2(&b.map(|v| Complex::from(v - mb)), false);
    let r = fa.zip_map(&fb, |p, q| {
        let c = p * q.conj();
        c / (c.abs() + 1e-6)
    });
    fft2(&r, true).map(|z| z.re)
}

/// The position `(p, q)` of the largest value (given).
pub fn argmax(c: &Image<f32>) -> (usize, usize) {
    let mut best = (0, 0);
    for (x, y) in c.coords() {
        if c[(x, y)] > c[best] {
            best = (x, y);
        }
    }
    best
}

/// C1. The four shifts that a circular peak at `(p, q)` can mean for tiles of `w × h`:
/// `tx` is `p` or `p − w`, `ty` is `q` or `q − h` (slide "A peak means four shifts").
///
/// Hint: convert to `isize` first, so that `p - w` may be negative.
pub fn candidates(p: usize, q: usize, w: usize, h: usize) -> [(isize, isize); 4] {
    todo!("C1: candidates")
}

/// C2. NCC of tile B placed at shift `(tx, ty)` on tile A (B's pixel `(x, y)` on A's pixel
/// `(x + tx, y + ty)`), over their overlap; returns `(ncc, number of overlapping pixels)`.
/// NCC is 0 for an empty overlap or a flat one (the matching unit).
///
/// Hint: as `ncc_2d` of the matching unit: the sums of f, g, f², g² and fg over the overlap
/// (`a.contains(ax, ay)`), in `f64`.
pub fn ncc_overlap(a: &Image<f32>, b: &Image<f32>, tx: isize, ty: isize) -> (f32, usize) {
    todo!("C2: ncc_overlap")
}

/// C3. The shift of tile B against tile A: the highest peak of their phase correlation, its four
/// candidates, and of those with at least `min_overlap` pixels of overlap the one with the
/// best NCC (slide "Resolving the candidates"). Returns `(tx, ty, ncc)`, or `None` if no
/// candidate has enough overlap.
///
/// Hint: `argmax(&phase_correlation(a, b))`, then `candidates`, then `ncc_overlap` for each.
pub fn register_pair(a: &Image<f32>, b: &Image<f32>, min_overlap: usize) -> Option<(isize, isize, f32)> {
    todo!("C3: register_pair")
}

/// C4. The sub-pixel offset of a peak from three samples, the peak `c0` and its neighbours
/// `cm` (at −1) and `cp` (at +1): the top of the parabola through them (slide "The math:
/// sub-pixel peaks"). 0 if the three lie on a line.
///
/// Hint: one line, plus the check for a zero denominator.
pub fn parabola(cm: f32, c0: f32, cp: f32) -> f32 {
    todo!("C4: parabola")
}

/// A measured shift between two tiles: tile `j` sits at `(dx, dy)` from tile `i`, with the
/// weight `w`.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub i: usize,
    pub j: usize,
    pub dx: f64,
    pub dy: f64,
    pub w: f64,
}

/// C5. Global alignment: the positions of `n` tiles that agree best with all measured shifts,
/// minimising `sum of w (x_j − x_i − d)²` separately in x and in y, with tile 0 fixed at
/// (0, 0) (slide "The math: least squares on a graph"). Build the weighted graph Laplacian `L`
/// and the right-hand side `b` for tiles 1..n−1 and solve `L x = b`. Returns `None` if the
/// system cannot be solved (a tile without any pair).
///
/// Hint: a matrix `vec![vec![0.0; n - 1]; n - 1]` (row `k - 1` belongs to tile `k`). For each
/// pair: add `w` to the diagonal entries of `i` and `j`, subtract `w` at `(i, j)` and `(j, i)`, add
/// `w · d` to `b` of `j` and subtract it from `b` of `i`, leaving out everything of tile 0.
/// Then `linalg::solve(&l, &bx)` (it returns an `Option`; `?` passes `None` on).
pub fn global_positions(n: usize, pairs: &[Pair]) -> Option<Vec<(f64, f64)>> {
    todo!("C5: global_positions")
}
