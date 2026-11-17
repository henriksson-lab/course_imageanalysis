//! Helpers shared by the units of day 11 (not exercises): matrix-vector products for the
//! ℓ1 units, and drawing 1D signals for their programs (`run.rs`).

use imgcourse::Image;

/// `A x`.
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(r, v)| r * v).sum()).collect()
}

/// `Aᵀ r`.
pub fn mat_t_vec(a: &[Vec<f64>], r: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a[0].len()];
    for (row, ri) in a.iter().zip(r) {
        for (o, aij) in out.iter_mut().zip(row) {
            *o += aij * ri;
        }
    }
    out
}

/// How `plot` draws a series.
#[derive(Clone, Copy, PartialEq)]
pub enum Style {
    /// A connected line.
    Line,
    /// A vertical line from 0 to each nonzero value.
    Stems,
    /// A dot (5 × 5 pixels) at each nonzero value.
    Dots,
}

/// Draw signals into a picture, `scale` pixels per sample and `height` pixels high: each
/// series is (values, colour, style), drawn in that order. 0 is the thin grey horizontal line.
pub fn plot(series: &[(&[f64], [f32; 3], Style)], scale: usize, height: usize) -> Image<[f32; 3]> {
    let n = series[0].0.len();
    let lo = series.iter().flat_map(|s| s.0.iter()).fold(0.0f64, |m, v| m.min(*v));
    let hi = series.iter().flat_map(|s| s.0.iter()).fold(0.0f64, |m, v| m.max(*v));
    let row = |v: f64| ((hi - v) / (hi - lo + 1e-12) * (height - 9) as f64) as usize + 4;
    let mut pic = Image::new(n * scale, height, [1.0f32; 3]);
    for x in 0..n * scale {
        pic[(x, row(0.0))] = [0.85; 3];
    }
    let vline = |pic: &mut Image<[f32; 3]>, x: usize, r0: usize, r1: usize, c: [f32; 3]| {
        for y in r0.min(r1)..=r0.max(r1) {
            pic[(x, y)] = c;
        }
    };
    for &(values, colour, style) in series {
        for i in 0..n {
            let x = i * scale + scale / 2;
            match style {
                Style::Stems if values[i] != 0.0 => vline(&mut pic, x, row(0.0), row(values[i]), colour),
                Style::Dots if values[i] != 0.0 => {
                    for dy in -2..=2isize {
                        for dx in -2..=2isize {
                            let (px, py) = (x as isize + dx, row(values[i]) as isize + dy);
                            if pic.contains(px, py) {
                                pic[(px as usize, py as usize)] = colour;
                            }
                        }
                    }
                }
                Style::Line if i + 1 < n => {
                    for s in 0..scale {
                        let f = |s: usize| values[i] + (values[i + 1] - values[i]) * s as f64 / scale as f64;
                        vline(&mut pic, x + s, row(f(s)), row(f(s + 1)), colour);
                    }
                }
                _ => {}
            }
        }
    }
    pic
}
