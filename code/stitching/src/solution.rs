// Model solution for the coding exercises of "From a peak to a mosaic".

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
pub fn candidates(p: usize, q: usize, w: usize, h: usize) -> [(isize, isize); 4] {
    let (p, q, w, h) = (p as isize, q as isize, w as isize, h as isize);
    [(p, q), (p - w, q), (p, q - h), (p - w, q - h)]
}

/// C2. NCC of tile B placed at shift `(tx, ty)` on tile A (B's pixel `(x, y)` on A's pixel
/// `(x + tx, y + ty)`), over their overlap; returns `(ncc, number of overlapping pixels)`.
/// NCC is 0 for an empty overlap or a flat one (the matching unit).
pub fn ncc_overlap(a: &Image<f32>, b: &Image<f32>, tx: isize, ty: isize) -> (f32, usize) {
    let (mut n, mut sf, mut sg, mut sff, mut sgg, mut sfg) = (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in b.coords() {
        let (ax, ay) = (x as isize + tx, y as isize + ty);
        if a.contains(ax, ay) {
            let (f, g) = (a[(ax as usize, ay as usize)] as f64, b[(x, y)] as f64);
            n += 1;
            sf += f;
            sg += g;
            sff += f * f;
            sgg += g * g;
            sfg += f * g;
        }
    }
    if n == 0 {
        return (0.0, 0);
    }
    let nn = n as f64;
    let (vf, vg, cov) = (sff - sf * sf / nn, sgg - sg * sg / nn, sfg - sf * sg / nn);
    let ncc = if vf > 1e-12 && vg > 1e-12 { cov / (vf * vg).sqrt() } else { 0.0 };
    (ncc as f32, n)
}

/// C3. The shift of tile B against tile A: the highest peak of their phase correlation, its four
/// candidates, and of those with at least `min_overlap` pixels of overlap the one with the
/// best NCC (slide "Resolving the candidates"). Returns `(tx, ty, ncc)`, or `None` if no
/// candidate has enough overlap.
pub fn register_pair(a: &Image<f32>, b: &Image<f32>, min_overlap: usize) -> Option<(isize, isize, f32)> {
    let (p, q) = argmax(&phase_correlation(a, b));
    let mut best: Option<(isize, isize, f32)> = None;
    for (tx, ty) in candidates(p, q, a.width(), a.height()) {
        let (c, n) = ncc_overlap(a, b, tx, ty);
        if n >= min_overlap && best.is_none_or(|bst| c > bst.2) {
            best = Some((tx, ty, c));
        }
    }
    best
}

/// C4. The sub-pixel offset of a peak from three samples, the peak `c0` and its neighbours
/// `cm` (at −1) and `cp` (at +1): the top of the parabola through them (slide "The math:
/// sub-pixel peaks"). 0 if the three lie on a line.
pub fn parabola(cm: f32, c0: f32, cp: f32) -> f32 {
    let denom = 2.0 * (cm - 2.0 * c0 + cp);
    if denom == 0.0 { 0.0 } else { (cm - cp) / denom }
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
pub fn global_positions(n: usize, pairs: &[Pair]) -> Option<Vec<(f64, f64)>> {
    let m = n - 1; // unknowns: tiles 1..n-1 (tile 0 is fixed at 0)
    let mut l = vec![vec![0.0; m]; m];
    let (mut bx, mut by) = (vec![0.0; m], vec![0.0; m]);
    for p in pairs {
        // the pair adds w to the diagonal of both tiles and −w off the diagonal;
        // its shift d pulls tile j by +w d and tile i by −w d
        for (k, other, sign) in [(p.j, p.i, 1.0), (p.i, p.j, -1.0)] {
            if k == 0 {
                continue;
            }
            l[k - 1][k - 1] += p.w;
            if other != 0 {
                l[k - 1][other - 1] -= p.w;
            }
            bx[k - 1] += sign * p.w * p.dx;
            by[k - 1] += sign * p.w * p.dy;
        }
    }
    let xs = linalg::solve(&l, &bx)?;
    let ys = linalg::solve(&l, &by)?;
    let mut pos = vec![(0.0, 0.0)];
    pos.extend(xs.into_iter().zip(ys));
    Some(pos)
}
