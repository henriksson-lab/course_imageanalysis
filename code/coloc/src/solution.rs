// Model solution for the coding exercises of "Colocalisation of two channels".

use imgcourse::Image;

// All functions take the two channels `a`, `b` and a `mask`: only pixels where the mask is
// `true` (the region of interest) count.

/// C1. Pearson's correlation of the two channels over the pixels of the mask
/// (slide "The math: correlation and overlap").
pub fn pearson(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut n, mut sa, mut sb) = (0.0f64, 0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        n += 1.0;
        sa += a[p] as f64;
        sb += b[p] as f64;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        let (da, db) = (a[p] as f64 - ma, b[p] as f64 - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

/// C2. The overlap coefficient and Manders' k1, k2: with the uncentred sums
/// S_AB = Σ A·B, S_AA = Σ A², S_BB = Σ B², returns
/// `(S_AB / sqrt(S_AA · S_BB), S_AB / S_AA, S_AB / S_BB)`.
pub fn overlap(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> (f32, f32, f32) {
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        let (va, vb) = (a[p] as f64, b[p] as f64);
        sab += va * vb;
        saa += va * va;
        sbb += vb * vb;
    }
    ((sab / (saa * sbb).sqrt()) as f32, (sab / saa) as f32, (sab / sbb) as f32)
}

/// C3. Manders' M1 and M2 with thresholds `ta`, `tb` ("above" means strictly greater):
/// M1 = Σ A over pixels above in both / Σ A over pixels above in A, and M2 the same for B.
pub fn manders(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, ta: f32, tb: f32) -> (f32, f32) {
    let (mut a_both, mut a_above, mut b_both, mut b_above) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        let (va, vb) = (a[p] as f64, b[p] as f64);
        let (above_a, above_b) = (a[p] > ta, b[p] > tb);
        if above_a {
            a_above += va;
            if above_b {
                a_both += va;
            }
        }
        if above_b {
            b_above += vb;
            if above_a {
                b_both += vb;
            }
        }
    }
    ((a_both / a_above) as f32, (b_both / b_above) as f32)
}

/// C4a. The ranks of a list of values: 1 for the smallest, n for the largest; values that
/// are equal share the average of their ranks (two values tied for ranks 1 and 2 both get 1.5).
pub fn ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let mut rank = vec![0.0; n];
    let mut start = 0;
    while start < n {
        // the group of equal values is order[start..end]
        let mut end = start + 1;
        while end < n && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // ranks start + 1 ..= end, their average
        let average = (start + 1 + end) as f32 / 2.0;
        for &i in &order[start..end] {
            rank[i] = average;
        }
        start = end;
    }
    rank
}

/// C4b. Spearman's coefficient: Pearson's r of the ranks of the pixels in the mask
/// (slide "Rank and robustness").
pub fn spearman(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    let inside: Vec<(usize, usize)> = a.coords().filter(|&p| mask[p]).collect();
    let va: Vec<f32> = inside.iter().map(|&p| a[p]).collect();
    let vb: Vec<f32> = inside.iter().map(|&p| b[p]).collect();
    let (ra, rb) = (ranks(&va), ranks(&vb));
    let mut rank_a = Image::new(a.width(), a.height(), 0.0f32);
    let mut rank_b = Image::new(a.width(), a.height(), 0.0f32);
    for (k, &p) in inside.iter().enumerate() {
        rank_a[p] = ra[k];
        rank_b[p] = rb[k];
    }
    pearson(&rank_a, &rank_b, mask)
}

/// C5a. The orthogonal regression line B = slope · A + intercept through the scatter plot
/// (slide "The math: the regression line"): with the centred sums of squares and products
/// s_AA, s_BB, s_AB, slope = (s_BB − s_AA + sqrt((s_BB − s_AA)² + 4 s_AB²)) / (2 s_AB).
/// Returns `(slope, intercept)`.
pub fn orthogonal_line(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> (f32, f32) {
    let (mut n, mut sa, mut sb) = (0.0f64, 0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        n += 1.0;
        sa += a[p] as f64;
        sb += b[p] as f64;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        let (da, db) = (a[p] as f64 - ma, b[p] as f64 - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    let slope = (sbb - saa + ((sbb - saa).powi(2) + 4.0 * sab * sab).sqrt()) / (2.0 * sab);
    (slope as f32, (mb - slope * ma) as f32)
}

/// C5b. Costes' thresholds (slide "Thresholds: Costes' method"): start t_A at the largest A
/// in the mask and lower it in `steps` equal steps down to the smallest; t_B = slope · t_A +
/// intercept follows the line. Stop at the first step where Pearson's r of the pixels
/// **below** the thresholds (A ≤ t_A or B ≤ t_B, inside the mask) is ≤ 0, and return
/// `Some((t_A, t_B))` there (or at the last step). `None` if r of the whole mask is ≤ 0: then
/// there is no rising line to walk down.
pub fn costes(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, steps: usize) -> Option<(f32, f32)> {
    if pearson(a, b, mask) <= 0.0 {
        return None;
    }
    let (slope, intercept) = orthogonal_line(a, b, mask);
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in a.coords().filter(|&p| mask[p]) {
        lo = lo.min(a[p]);
        hi = hi.max(a[p]);
    }
    let step = (hi - lo) / steps as f32;
    let (mut ta, mut tb) = (hi, slope * hi + intercept);
    for k in 1..=steps {
        ta = hi - k as f32 * step;
        tb = slope * ta + intercept;
        let below = Image::from_fn(a.width(), a.height(), |x, y| mask[(x, y)] && (a[(x, y)] <= ta || b[(x, y)] <= tb));
        if pearson(a, b, &below) <= 0.0 {
            break;
        }
    }
    Some((ta, tb))
}
