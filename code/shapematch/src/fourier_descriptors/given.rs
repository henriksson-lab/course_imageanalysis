//! Given: tracing the outline of a mask, filling a polygon, test shapes, the three simple moves
//! of an outline (new start, mirror image, similarity) and drawing. Not an exercise here.
//!
//! An outline is a `Vec<Complex>`: the point `(x, y)` is `x + i·y`, with `y` pointing down, so
//! multiplying by `e^{iα}` turns a point clockwise on screen.

use imgcourse::{Complex, Image};

/// The best fit of outline A onto outline B: `B_n ≈ λ·A_{n+m} + t` (with A replaced by its
/// mirror image if `mirrored`).
#[derive(Clone, Copy, Debug)]
pub struct Match {
    /// The starting point: sample `m` of A goes onto sample 0 of B.
    pub m: usize,
    /// `λ = s·e^{iα}`: scale `s = |λ|` and angle `α = arg λ` (clockwise on screen).
    pub lambda: Complex,
    /// The translation `t = c^B_0 − λ·c^A_0`.
    pub t: Complex,
    /// `|q(m)| / √(Σ|c^A_k|² Σ|c^B_k|²)`: 1 for a perfect fit.
    pub score: f32,
    /// True if the mirror image of A fits better than A.
    pub mirrored: bool,
}

impl Match {
    pub fn scale(&self) -> f32 {
        self.lambda.abs()
    }
    /// The angle in degrees, clockwise on screen.
    pub fn angle_deg(&self) -> f32 {
        self.lambda.arg().to_degrees()
    }
    /// `√(1 − score²)`: the rms distance left after the best fit, relative to the size of B.
    pub fn misfit(&self) -> f32 {
        (1.0 - self.score * self.score).max(0.0).sqrt()
    }
}

/// The signed frequency of index `j` in a DFT of length `n`: `j` for `j < n/2`, else `j − n`.
pub fn freq(j: usize, n: usize) -> isize {
    if 2 * j < n { j as isize } else { j as isize - n as isize }
}

/// `c_k` for a signed `k` (indices wrap around modulo the length).
pub fn coef(c: &[Complex], k: isize) -> Complex {
    c[k.rem_euclid(c.len() as isize) as usize]
}

/// Start `m` samples later: `z'_n = z_{n+m}`.
pub fn restart(z: &[Complex], m: usize) -> Vec<Complex> {
    (0..z.len()).map(|n| z[(n + m) % z.len()]).collect()
}

/// The mirror image (`y → −y`) walked clockwise again from the same start: `z'_n = conj(z_{−n})`.
pub fn mirror_traced(z: &[Complex]) -> Vec<Complex> {
    let n = z.len();
    (0..n).map(|i| z[(n - i) % n].conj()).collect()
}

/// `z ↦ s·e^{iα}·z + t` (angle in degrees, clockwise on screen).
pub fn similarity(z: &[Complex], s: f32, alpha_deg: f32, t: Complex) -> Vec<Complex> {
    let l = Complex::from_polar(s, alpha_deg.to_radians());
    z.iter().map(|&p| l * p + t).collect()
}

/// Outline A moved onto outline B by a match: `λ·A_{n+m} + t` (mirrored first if the match says so).
pub fn aligned(a: &[Complex], fit: &Match) -> Vec<Complex> {
    let src = if fit.mirrored { mirror_traced(a) } else { a.to_vec() };
    restart(&src, fit.m).iter().map(|&p| fit.lambda * p + fit.t).collect()
}

/// `n` points equally spaced in vertex INDEX along the closed polygon (every edge counts as one
/// step, whatever its length): what you get without arc-length resampling.
pub fn resample_by_index(poly: &[Complex], n: usize) -> Vec<Complex> {
    let m = poly.len();
    (0..n)
        .map(|i| {
            let t = (m * i) as f32 / n as f32;
            let j = t.floor() as usize;
            let f = t - j as f32;
            poly[j % m] * (1.0 - f) + poly[(j + 1) % m] * f
        })
        .collect()
}

/// The outer boundary of the object in `mask` that holds the first `true` pixel in raster order,
/// as pixel centres `x + i·y`, walked clockwise on screen (Moore tracing, 8-connected;
/// the labelling day's boundary walk).
pub fn trace_boundary(mask: &Image<bool>) -> Vec<Complex> {
    let (w, h) = (mask.width() as isize, mask.height() as isize);
    let fg = |x: isize, y: isize| x >= 0 && y >= 0 && x < w && y < h && mask[(x as usize, y as usize)];
    // the 8 neighbours clockwise on screen, starting east
    const D: [(isize, isize); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
    let Some(start) = mask.coords().position(|(x, y)| mask[(x, y)]) else { return vec![] };
    let (sx, sy) = (start as isize % w, start as isize / w);
    // from (x, y) with background in direction b: scan clockwise from b, step to the first
    // object pixel; the new b points to the last background pixel seen, from the new pixel
    let step = |x: isize, y: isize, b: usize| -> Option<(isize, isize, usize)> {
        for j in 1..=8 {
            let d = (b + j) % 8;
            let (nx, ny) = (x + D[d].0, y + D[d].1);
            if fg(nx, ny) {
                let pd = (b + j - 1) % 8;
                let (px, py) = (x + D[pd].0, y + D[pd].1);
                let nb = (0..8).find(|&q| nx + D[q].0 == px && ny + D[q].1 == py).unwrap_or(0);
                return Some((nx, ny, nb));
            }
        }
        None
    };
    let mut pts = vec![Complex::new(sx as f32, sy as f32)];
    let Some(first) = step(sx, sy, 4) else { return pts };
    let (mut x, mut y, mut b) = first;
    loop {
        if (x, y) == (sx, sy) {
            let nxt = step(x, y, b).unwrap();
            if (nxt.0, nxt.1) == (first.0, first.1) {
                break;
            }
        }
        pts.push(Complex::new(x as f32, y as f32));
        (x, y, b) = step(x, y, b).unwrap();
    }
    pts
}

/// A `w × h` mask of the pixels whose centre `(x, y)` lies inside the closed polygon.
pub fn fill_polygon(poly: &[Complex], w: usize, h: usize) -> Image<bool> {
    Image::from_fn(w, h, |x, y| {
        let (px, py) = (x as f32, y as f32);
        let mut inside = false;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            if (a.im > py) != (b.im > py) && px < a.re + (py - a.im) / (b.im - a.im) * (b.re - a.re) {
                inside = !inside;
            }
        }
        inside
    })
}

/// The letter F as a polygon (as `samples::letter_f`), centred, about 1.4 high, clockwise on screen.
pub fn letter_f() -> Vec<Complex> {
    let c = |u: f32, v: f32| Complex::new(2.0 * (u - 0.47), 2.0 * (v - 0.5));
    vec![
        c(0.25, 0.15), c(0.75, 0.15), c(0.75, 0.30), c(0.40, 0.30), c(0.40, 0.45), c(0.62, 0.45),
        c(0.62, 0.58), c(0.40, 0.58), c(0.40, 0.85), c(0.25, 0.85),
    ]
}

/// A smooth blob `r(θ) = 1 + Σ a_j cos(j θ + φ_j)` for the given `(j, a_j, φ_j)`, 360 points,
/// clockwise on screen. Five such blobs make the "which two are the same shape?" catalogue.
pub fn blob(terms: &[(f32, f32, f32)]) -> Vec<Complex> {
    (0..360)
        .map(|i| {
            let th = (i as f32).to_radians();
            let r = 1.0 + terms.iter().map(|&(j, a, p)| a * (j * th + p).cos()).sum::<f32>();
            Complex::from_polar(r, th)
        })
        .collect()
}

/// The traced outline of a polygon rasterised at a pose: scaled by `s` pixels per unit, turned
/// by `alpha_deg`, centred in a square mask with a margin.
pub fn raster_outline(poly: &[Complex], s: f32, alpha_deg: f32) -> (Image<bool>, Vec<Complex>) {
    let r = poly.iter().map(|p| p.abs()).fold(0.0, f32::max);
    let w = (2.0 * r * s).ceil() as usize + 8;
    let c = (w as f32 - 1.0) / 2.0;
    let posed = similarity(poly, s, alpha_deg, Complex::new(c, c));
    let mask = fill_polygon(&posed, w, w);
    let outline = trace_boundary(&mask);
    (mask, outline)
}

/// Draws a closed outline into an RGB picture (nearest pixel, short line segments).
pub fn draw_outline(img: &mut Image<[f32; 3]>, z: &[Complex], colour: [f32; 3]) {
    for i in 0..z.len() {
        let (a, b) = (z[i], z[(i + 1) % z.len()]);
        let steps = ((b - a).abs() * 2.0).ceil().max(1.0) as usize;
        for s in 0..=steps {
            let p = a + (b - a) * (s as f32 / steps as f32);
            let (x, y) = (p.re.round(), p.im.round());
            if x >= 0.0 && y >= 0.0 && (x as usize) < img.width() && (y as usize) < img.height() {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
}
