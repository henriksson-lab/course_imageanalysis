// Model solution for the coding exercises of "Bleed-through and compensation".

use imgcourse::Image;

/// A 2 × 2 matrix, row by row: `m[row][column]`; rows = channels, columns = dyes.
pub type Mat2 = [[f32; 2]; 2];

/// C1a. The value a channel records for one unit of a dye: the dye's emission spectrum times
/// the filter's transmission, added over the wavelength bins, times the bin width `dl`.
pub fn channel_value(spectrum: &[f32], filter: &[f32], dl: f32) -> f32 {
    let mut sum = 0.0;
    for i in 0..spectrum.len() {
        sum += spectrum[i] * filter[i];
    }
    sum * dl
}

/// C1b. The mixing matrix of two dyes and two channels: entry `[c][k]` is what channel c
/// records for one unit of dye k. Column k is dye k as the channels see it.
pub fn mixing_matrix(dyes: [&[f32]; 2], filters: [&[f32]; 2], dl: f32) -> Mat2 {
    let mut m = [[0.0; 2]; 2];
    for c in 0..2 {
        for k in 0..2 {
            m[c][k] = channel_value(dyes[k], filters[c], dl);
        }
    }
    m
}

/// C2. The slope of the line through the origin that fits the points (x_i, y_i) best in the
/// least-squares sense: `Σ x y / Σ x²` (a single-stained control gives a spillover).
pub fn slope_through_origin(xs: &[f32], ys: &[f32]) -> f32 {
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        sxy += x * y;
        sxx += x * x;
    }
    sxy / sxx
}

/// C3a. The inverse of a 2 × 2 matrix by the formula of the slide "The math: the 2 × 2 inverse".
pub fn inverse2(m: Mat2) -> Mat2 {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
}

/// C3b. Compensate a two-channel image: at every pixel, the dye amounts `a = S⁻¹ y` from the
/// channel values `y = (ch1, ch2)`. Returns the two dye images.
pub fn compensate(ch1: &Image<f32>, ch2: &Image<f32>, s: Mat2) -> (Image<f32>, Image<f32>) {
    let inv = inverse2(s);
    let a1 = ch1.zip_map(ch2, |y1, y2| inv[0][0] * y1 + inv[0][1] * y2);
    let a2 = ch1.zip_map(ch2, |y1, y2| inv[1][0] * y1 + inv[1][1] * y2);
    (a1, a2)
}

/// C4. Pearson's correlation of two images over the pixels where `mask` is true:
/// `Σ (a − ā)(b − b̄) / sqrt(Σ (a − ā)² · Σ (b − b̄)²)`, with the means over the mask.
pub fn pearson(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut n, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for (x, y) in a.coords() {
        if mask[(x, y)] {
            n += 1.0;
            sa += a[(x, y)];
            sb += b[(x, y)];
        }
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.coords() {
        if mask[(x, y)] {
            let (da, db) = (a[(x, y)] - ma, b[(x, y)] - mb);
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
    }
    sab / (saa * sbb).sqrt()
}
