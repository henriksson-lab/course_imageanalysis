// Model solution for the coding exercises of "When one threshold is not enough".
//
// Objects are dark in this unit: a pixel is object where I(x) < T(x). Pixels outside the
// image are read by copying the edge pixel (Border::Clamp).

use imgcourse::{Border, Image};

/// A rule that turns the local mean `m` and standard deviation `s` into a threshold.
#[derive(Clone, Copy, Debug)]
pub enum Rule {
    /// T = m − C
    MeanC(f32),
    /// T = m + k s (Niblack; k is about −0.2)
    Niblack(f32),
    /// T = m (1 + k (s / R − 1)) (Sauvola; k about 0.2–0.5, R the largest possible s:
    /// 128 for grey levels 0–255, 0.5 for grey values 0–1)
    Sauvola { k: f32, range: f32 },
}

/// C1. The local mean and standard deviation in the (2r + 1) × (2r + 1) window around every
/// pixel: m = box(I), s = sqrt(box(I²) − m²) (slide "Local mean and variance are box filters").
/// The box sums are made in two passes, first along the rows, then along the columns.
pub fn local_mean_std(img: &Image<f32>, r: usize) -> (Image<f32>, Image<f32>) {
    let r = r as isize;
    let n = ((2 * r + 1) * (2 * r + 1)) as f64;
    // pass 1: sums of I and I² along each row
    let rows = img.map_xy(|x, y, _| {
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for d in -r..=r {
            let v = img.get_border(x as isize + d, y as isize, Border::Clamp) as f64;
            s1 += v;
            s2 += v * v;
        }
        (s1, s2)
    });
    // pass 2: sum the row sums along each column
    let mut mean = Image::new(img.width(), img.height(), 0.0);
    let mut std = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (mut s1, mut s2) = (0.0, 0.0);
        for d in -r..=r {
            let yy = (y as isize + d).clamp(0, img.height() as isize - 1) as usize;
            s1 += rows[(x, yy)].0;
            s2 += rows[(x, yy)].1;
        }
        let m = s1 / n;
        mean[(x, y)] = m as f32;
        std[(x, y)] = (s2 / n - m * m).max(0.0).sqrt() as f32; // max(0): rounding can make it slightly negative
    }
    (mean, std)
}

/// C2. The threshold of a rule for one pixel, from its local mean `m` and standard deviation `s`.
pub fn rule_threshold(rule: Rule, m: f32, s: f32) -> f32 {
    match rule {
        Rule::MeanC(c) => m - c,
        Rule::Niblack(k) => m + k * s,
        Rule::Sauvola { k, range } => m * (1.0 + k * (s / range - 1.0)),
    }
}

/// C3. A local threshold: object (true) where the pixel is darker than the threshold of its
/// own window, I(x) < T(x).
pub fn local_threshold(img: &Image<f32>, r: usize, rule: Rule) -> Image<bool> {
    let (mean, std) = local_mean_std(img, r);
    img.map_xy(|x, y, v| v < rule_threshold(rule, mean[(x, y)], std[(x, y)]))
}

/// C4. Hysteresis for **bright** structures: seeds are the pixels above `high`; keep every
/// pixel above `low` that is connected to a seed by a chain of neighbours above `low`
/// (4-neighbours, or 8 if `eight`).
pub fn hysteresis(img: &Image<f32>, low: f32, high: f32, eight: bool) -> Image<bool> {
    let mut kept = Image::new(img.width(), img.height(), false);
    // a stack of pixels whose neighbours still have to be looked at
    let mut stack = Vec::new();
    for (x, y) in img.coords() {
        if img[(x, y)] > high {
            kept[(x, y)] = true;
            stack.push((x, y));
        }
    }
    let steps: &[(isize, isize)] = if eight {
        &[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    } else {
        &[(0, -1), (-1, 0), (1, 0), (0, 1)]
    };
    while let Some((x, y)) = stack.pop() {
        for &(dx, dy) in steps {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if img.contains(nx, ny) {
                let (nx, ny) = (nx as usize, ny as usize);
                if !kept[(nx, ny)] && img[(nx, ny)] > low {
                    kept[(nx, ny)] = true;
                    stack.push((nx, ny));
                }
            }
        }
    }
    kept
}
