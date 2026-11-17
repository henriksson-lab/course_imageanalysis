// Model solution for the coding exercises of "Self-supervised denoising".

use imgcourse::{linalg, Border, Image};

/// A linear denoising filter: `out(p) = Σ weights[k] · in(p + offset k) + bias` over the
/// `(2 radius + 1)²` window around `p`, row by row. A blind-spot filter has centre weight 0.
#[derive(Clone, Debug)]
pub struct Filter {
    pub radius: usize,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Filter {
    /// The weight of the offset `(dx, dy)` from the centre.
    pub fn weight(&self, dx: isize, dy: isize) -> f32 {
        let n = 2 * self.radius as isize + 1;
        self.weights[((dy + self.radius as isize) * n + dx + self.radius as isize) as usize]
    }
}

/// C1. The grey values of the `(2 radius + 1)²` window around `(x, y)`, row by row (mirror
/// border). With `blind = true` the centre pixel itself is left out (one value fewer).
pub fn neighbourhood(img: &Image<f32>, x: usize, y: usize, radius: usize, blind: bool) -> Vec<f32> {
    let r = radius as isize;
    let mut v = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dy in -r..=r {
        for dx in -r..=r {
            if blind && dx == 0 && dy == 0 {
                continue;
            }
            v.push(img.get_border(x as isize + dx, y as isize + dy, Border::Mirror));
        }
    }
    v
}

/// C2. Train a linear filter by least squares: for every pixel, the prediction from the
/// window of `input` around it (without the centre if `blind`) should match `target` at
/// that pixel. One row per pixel: the window values and a 1 (for the bias).
/// Noise2Void: `input` = `target` = the noisy image, `blind = true`.
/// Noise2Noise: `input` = one noisy image, `target` = a second noisy image of the same thing.
pub fn fit_filter(input: &Image<f32>, target: &Image<f32>, radius: usize, blind: bool) -> Filter {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    for (x, y) in input.coords() {
        let mut row: Vec<f64> = neighbourhood(input, x, y, radius, blind).iter().map(|&v| v as f64).collect();
        row.push(1.0);
        rows.push(row);
        b.push(target[(x, y)] as f64);
    }
    let w = linalg::least_squares(&rows, &b).expect("the window values are linearly dependent");
    let n = (2 * radius + 1) * (2 * radius + 1);
    let mut weights = Vec::with_capacity(n);
    let mut k = 0;
    for i in 0..n {
        if blind && i == n / 2 {
            weights.push(0.0); // the blind spot
        } else {
            weights.push(w[k] as f32);
            k += 1;
        }
    }
    Filter { radius, weights, bias: w[k] as f32 }
}

/// C3. Apply the filter: every output pixel is the weighted sum of the window around it
/// (mirror border) plus the bias.
pub fn apply_filter(img: &Image<f32>, f: &Filter) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let window = neighbourhood(img, x, y, f.radius, false);
        f.bias + window.iter().zip(&f.weights).map(|(v, w)| v * w).sum::<f32>()
    })
}

/// C4. Put the pixel's own value back. The blind-spot prediction `pred` and the noisy pixel
/// are two independent estimates of the clean value; with the noise level `sigma` known
/// (from a calibration of the camera), mix them: `out = pred + c · (noisy − pred)` with
/// `c = 1 − sigma² / mean((noisy − pred)²)` (not below 0). Returns the image and `c`.
pub fn mix_back(noisy: &Image<f32>, pred: &Image<f32>, sigma: f32) -> (Image<f32>, f32) {
    let n = noisy.pixels().len() as f32;
    let residual2: f32 = noisy.pixels().iter().zip(pred.pixels()).map(|(y, p)| (y - p).powi(2)).sum::<f32>() / n;
    let c = (1.0 - sigma * sigma / residual2).max(0.0);
    (pred.zip_map(noisy, |p, y| p + c * (y - p)), c)
}
