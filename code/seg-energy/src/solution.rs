// Model solution for the coding exercises of "Segmentation as energy minimisation".

use imgcourse::Image;

/// C1a. The data costs of the two labels at every pixel, from two Gaussians of equal width:
/// D(ℓ) = (I − μℓ)² / (2 s²) (slide "The data term from an intensity model"; the constant
/// is left out). Returns the images (D(0), D(1)).
pub fn data_costs(img: &Image<f32>, mu0: f32, mu1: f32, s: f32) -> (Image<f32>, Image<f32>) {
    let d0 = img.map(|v| (v - mu0).powi(2) / (2.0 * s * s));
    let d1 = img.map(|v| (v - mu1).powi(2) / (2.0 * s * s));
    (d0, d1)
}

/// C1b. The energy of a labelling (labels 0 and 1): the data cost of every pixel's label,
/// plus `lambda` times the number of 4-neighbour pairs with different labels (w = 1).
pub fn energy(labels: &Image<u32>, d0: &Image<f32>, d1: &Image<f32>, lambda: f32) -> f32 {
    let mut data = 0.0;
    let mut boundary = 0.0;
    for (x, y) in labels.coords() {
        data += if labels[(x, y)] == 1 { d1[(x, y)] } else { d0[(x, y)] };
        // count each pair once: the neighbour to the right and the one below
        if x + 1 < labels.width() && labels[(x, y)] != labels[(x + 1, y)] {
            boundary += 1.0;
        }
        if y + 1 < labels.height() && labels[(x, y)] != labels[(x, y + 1)] {
            boundary += 1.0;
        }
    }
    data + lambda * boundary
}

/// C2. Thresholding as the minimum for λ = 0: every pixel takes its cheaper label
/// (1 where D(1) < D(0)).
pub fn threshold(d0: &Image<f32>, d1: &Image<f32>) -> Image<u32> {
    d0.zip_map(d1, |a, b| if b < a { 1 } else { 0 })
}

/// C3. The exact minimum of the chain energy Σ D_k(L_k) + λ Σ [L_k ≠ L_k+1] by dynamic
/// programming (math slide "Dynamic programming on a chain"). Returns the best labelling
/// and its energy.
pub fn chain_dp(d0: &[f32], d1: &[f32], lambda: f32) -> (Vec<u32>, f32) {
    let n = d0.len();
    let d = [d0, d1];
    // c[k][l]: the smallest energy of pixels 0..=k with pixel k labelled l;
    // from[k][l]: the label of pixel k − 1 in that best labelling.
    let mut c = vec![[0.0f32; 2]; n];
    let mut from = vec![[0usize; 2]; n];
    c[0] = [d0[0], d1[0]];
    for k in 1..n {
        for l in 0..2 {
            let same = c[k - 1][l];
            let other = c[k - 1][1 - l] + lambda;
            if same <= other {
                c[k][l] = d[l][k] + same;
                from[k][l] = l;
            } else {
                c[k][l] = d[l][k] + other;
                from[k][l] = 1 - l;
            }
        }
    }
    // the best last label, then follow the remembered choices back
    let mut l = if c[n - 1][1] < c[n - 1][0] { 1 } else { 0 };
    let best = c[n - 1][l];
    let mut labels = vec![0u32; n];
    for k in (0..n).rev() {
        labels[k] = l as u32;
        l = from[k][l];
    }
    (labels, best)
}

/// C4. Dynamic programming on every row of an image separately: the exact minimum for each
/// row on its own, ignoring the boundaries between rows.
pub fn rows_dp(d0: &Image<f32>, d1: &Image<f32>, lambda: f32) -> Image<u32> {
    let mut out = Image::new(d0.width(), d0.height(), 0u32);
    for y in 0..d0.height() {
        let (labels, _) = chain_dp(d0.row(y), d1.row(y), lambda);
        for x in 0..d0.width() {
            out[(x, y)] = labels[x];
        }
    }
    out
}
