// Model solution for the coding exercises of "Total variation flow".

use imgcourse::Image;

/// What a gradient of size `s` costs (slide "Three smoothers, three energies").
#[derive(Clone, Copy, Debug)]
pub enum Penalty {
    /// `s^2 / 2`
    Heat,
    /// `s`
    Tv,
    /// `K^2 / 2 * ln(1 + s^2 / K^2)`, with the contrast parameter `K`
    PeronaMalik(f32),
}

/// C1. The energy of a row: the sum of the penalty of every neighbour difference.
pub fn energy_1d(row: &[f32], penalty: Penalty) -> f32 {
    let mut e = 0.0;
    for i in 0..row.len().saturating_sub(1) {
        let s = (row[i + 1] - row[i]).abs();
        e += match penalty {
            Penalty::Heat => s * s / 2.0,
            Penalty::Tv => s,
            Penalty::PeronaMalik(k) => k * k / 2.0 * (1.0 + s * s / (k * k)).ln(),
        };
    }
    e
}

/// C2. One explicit step of TV flow on a row: between every pair of neighbours the flow is
/// `d / sqrt(d^2 + eps^2)` (about the sign of the difference `d`: one unit of flow, however
/// large the step). Mirror border: nothing flows out at the ends. Stable for `dt <= eps / 2`.
pub fn tv_step_1d(row: &[f32], dt: f32, eps: f32) -> Vec<f32> {
    let n = row.len();
    let mut flow = vec![0.0; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        let d = row[i + 1] - row[i];
        flow[i] = d / (d * d + eps * eps).sqrt();
    }
    let mut out = row.to_vec();
    for i in 0..n {
        if i + 1 < n {
            out[i] += dt * flow[i];
        }
        if i > 0 {
            out[i] -= dt * flow[i - 1];
        }
    }
    out
}

/// C3. One explicit step of TV flow on an image, `u + dt * div(grad u / |grad u|)`
/// (slide "TV flow: level lines move by their curvature"). At every pixel take the
/// differences to the right and downwards, `(ux, uy)`, divide both by
/// `sqrt(ux^2 + uy^2 + eps^2)` to get a vector `(px, py)` of length below 1; then the change
/// is the divergence, `px(x, y) - px(x-1, y) + py(x, y) - py(x, y-1)`. Mirror border: no
/// difference, and so no flow, across the edge of the image. Stable for `dt <= eps / 4`.
pub fn tv_step(img: &Image<f32>, dt: f32, eps: f32) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    let mut px = Image::new(w, h, 0.0);
    let mut py = Image::new(w, h, 0.0);
    for (x, y) in img.coords() {
        let ux = if x + 1 < w { img[(x + 1, y)] - img[(x, y)] } else { 0.0 };
        let uy = if y + 1 < h { img[(x, y + 1)] - img[(x, y)] } else { 0.0 };
        let m = (ux * ux + uy * uy + eps * eps).sqrt();
        px[(x, y)] = ux / m;
        py[(x, y)] = uy / m;
    }
    let mut out = img.clone();
    for (x, y) in img.coords() {
        let left = if x > 0 { px[(x - 1, y)] } else { 0.0 };
        let up = if y > 0 { py[(x, y - 1)] } else { 0.0 };
        out[(x, y)] += dt * (px[(x, y)] - left + py[(x, y)] - up);
    }
    out
}
