//! Training data and rectangle features for the detector in `main.rs` (given, not an
//! exercise): windows of synthetic fluorescence images, centred on a nucleus (positive) or not
//! (negative), and a pool of rectangle features (the previous unit) on a 24 × 24 window.

use imgcourse::{samples, Image, Rng};

pub use crate::common::{integral, rect};

/// The side of the detection window.
pub const WINDOW: usize = 24;

/// The side of the synthetic images.
pub const SIZE: usize = 160;

/// One rectangle feature: `kind` 0 edge across, 1 edge down, 2 line across, 3 line down,
/// 4 diagonal, 5 centre − surround; `(x, y)` the top-left of the template in the window,
/// `w × h` one of its rectangles.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub kind: usize,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

const NAMES: [&str; 6] = ["edge across", "edge down", "line across", "line down", "diagonal", "centre − surround"];
/// Template size in rectangles, across and down, for each kind.
const CELLS: [(usize, usize); 6] = [(2, 1), (1, 2), (3, 1), (1, 3), (2, 2), (3, 3)];

impl Feature {
    pub fn name(&self) -> String {
        let (a, b) = CELLS[self.kind];
        format!("{} at ({}, {}), {} × {} px", NAMES[self.kind], self.x, self.y, a * self.w, b * self.h)
    }

    /// The feature's value for the window whose top-left pixel is `(x0, y0)` in an image with
    /// the integral images `s1` (of I) and `s2` (of I²): the rectangle sums, divided by the area
    /// of one rectangle and by the window's standard deviation (slide "Normalising for lighting").
    pub fn value(&self, s1: &Image<f64>, s2: &Image<f64>, x0: usize, y0: usize) -> f32 {
        let r = |dx: usize, dy: usize, w: usize, h: usize| rect(s1, x0 + self.x + dx, y0 + self.y + dy, w, h);
        let (w, h) = (self.w, self.h);
        let v = match self.kind {
            0 => r(0, 0, w, h) - r(w, 0, w, h),
            1 => r(0, 0, w, h) - r(0, h, w, h),
            2 => 2.0 * r(w, 0, w, h) - r(0, 0, w, h) - r(2 * w, 0, w, h),
            3 => 2.0 * r(0, h, w, h) - r(0, 0, w, h) - r(0, 2 * h, w, h),
            4 => r(0, 0, w, h) + r(w, h, w, h) - r(w, 0, w, h) - r(0, h, w, h),
            _ => r(w, h, w, h) - r(0, 0, 3 * w, 3 * h) / 9.0,
        };
        let n = (WINDOW * WINDOW) as f64;
        let m = rect(s1, x0, y0, WINDOW, WINDOW) / n;
        let sd = (rect(s2, x0, y0, WINDOW, WINDOW) / n - m * m).max(0.0).sqrt();
        (v / (w * h) as f64 / (sd + 0.01)) as f32
    }
}

/// All features with rectangles of 2, 4, 6 or 8 pixels per side, on a grid of 3 pixels.
pub fn feature_pool() -> Vec<Feature> {
    let mut pool = Vec::new();
    for (kind, &(a, b)) in CELLS.iter().enumerate() {
        for w in [2, 4, 6, 8] {
            for h in [2, 4, 6, 8] {
                if a * w > WINDOW || b * h > WINDOW {
                    continue;
                }
                for y in (0..=WINDOW - b * h).step_by(3) {
                    for x in (0..=WINDOW - a * w).step_by(3) {
                        pool.push(Feature { kind, x, y, w, h });
                    }
                }
            }
        }
    }
    pool
}

/// The centres of the nuclei in a label image (the mean position of each label's pixels).
pub fn centres(labels: &Image<u32>) -> Vec<(f32, f32)> {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut sum = vec![(0.0f32, 0.0f32, 0usize); n + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        sum[l].0 += x as f32;
        sum[l].1 += y as f32;
        sum[l].2 += 1;
    }
    sum[1..].iter().filter(|s| s.2 > 0).map(|s| (s.0 / s.2 as f32, s.1 / s.2 as f32)).collect()
}

/// A labelled set of windows: `windows[i] = (k, x, y)` is the window with top-left pixel
/// `(x, y)` in `images[k]`, and `labels[i]` is `+1` (nucleus) or `-1` (background).
pub struct Examples {
    pub images: Vec<Image<f32>>,
    pub windows: Vec<(usize, usize, usize)>,
    pub labels: Vec<f64>,
}

/// Windows from the synthetic images with the given seeds: every nucleus whose window fits
/// gives a positive window centred on it, shifted by up to one pixel. Every image also gives
/// twice as many negative windows: half of them at random places, half of them near misses,
/// whose centre is 6 to 12 px from a nucleus centre. No negative window is centred within 6 px
/// of a nucleus centre.
pub fn examples(seeds: &[u64]) -> Examples {
    let mut ex = Examples { images: Vec::new(), windows: Vec::new(), labels: Vec::new() };
    let half = WINDOW / 2;
    let max = (SIZE - WINDOW) as isize;
    for (k, &seed) in seeds.iter().enumerate() {
        let (img, lab) = samples::nuclei(SIZE, 12, seed);
        let mut rng = Rng::new(seed + 1000);
        let cs = centres(&lab);
        let mut positives = 0;
        for &(cx, cy) in &cs {
            let x = (cx.round() as isize + rng.below(3) as isize - 1 - half as isize).clamp(0, max) as usize;
            let y = (cy.round() as isize + rng.below(3) as isize - 1 - half as isize).clamp(0, max) as usize;
            if ((x + half) as f32 - cx).abs() <= 2.0 && ((y + half) as f32 - cy).abs() <= 2.0 {
                ex.windows.push((k, x, y));
                ex.labels.push(1.0);
                positives += 1;
            }
        }
        let mut negatives = 0;
        while negatives < 2 * positives {
            let (x, y) = if negatives % 2 == 0 {
                (rng.below(SIZE - WINDOW + 1), rng.below(SIZE - WINDOW + 1))
            } else {
                let (cx, cy) = cs[rng.below(cs.len())];
                let (r, a) = (6.0 + 6.0 * rng.uniform(), std::f32::consts::TAU * rng.uniform());
                let x = (cx + r * a.cos()).round() as isize - half as isize;
                let y = (cy + r * a.sin()).round() as isize - half as isize;
                (x.clamp(0, max) as usize, y.clamp(0, max) as usize)
            };
            let (mx, my) = ((x + half) as f32, (y + half) as f32);
            if cs.iter().all(|&(cx, cy)| (cx - mx).hypot(cy - my) >= 6.0) {
                ex.windows.push((k, x, y));
                ex.labels.push(-1.0);
                negatives += 1;
            }
        }
        ex.images.push(img);
    }
    ex
}
