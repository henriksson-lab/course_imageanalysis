// Model solution for the coding exercises of "Histograms and a global threshold".

use imgcourse::Image;

/// The bin of a grey value `v` in a histogram with `bins` equal bins over [0, 1]: bin `i`
/// holds the values from `i / bins` up to (not including) `(i + 1) / bins`. The value 1 goes
/// into the last bin, values outside [0, 1] into the first or last bin.
/// With 256 bins, an 8-bit grey level `k` (stored as `k / 255`) lands in bin `k`.
pub fn bin_of(v: f32, bins: usize) -> usize {
    let b = (v * bins as f32).floor();
    if b < 0.0 {
        0
    } else {
        (b as usize).min(bins - 1)
    }
}

/// C1. The histogram: `h[i]` = the number of pixels whose grey value falls into bin `i`.
pub fn histogram(img: &Image<f32>, bins: usize) -> Vec<usize> {
    let mut h = vec![0; bins];
    for (x, y) in img.coords() {
        h[bin_of(img[(x, y)], bins)] += 1;
    }
    h
}

/// C2. The cumulative histogram, normalised: `p[t]` = the fraction of pixels in bins `0..=t`
/// (the slide "The cumulative histogram and the percentile threshold"). The last value is 1.
pub fn cumulative(h: &[usize]) -> Vec<f32> {
    let n: usize = h.iter().sum();
    let mut p = Vec::with_capacity(h.len());
    let mut running = 0;
    for &count in h {
        running += count;
        p.push(running as f32 / n as f32);
    }
    p
}

/// C3. Threshold at bin `t`: object (true) where the pixel's bin is above `t`, background
/// (false) where it is `t` or below.
pub fn threshold(img: &Image<f32>, bins: usize, t: usize) -> Image<bool> {
    img.map(|v| bin_of(v, bins) > t)
}

/// C4. The percentile threshold: the smallest `t` that leaves at most the fraction `q` of the
/// pixels above `t` ("keep the brightest q").
pub fn percentile_threshold(h: &[usize], q: f32) -> usize {
    let n: usize = h.iter().sum();
    let allowed = q * n as f32;
    // the number of pixels above t, for t = 0, 1, 2, ...
    let mut above = n;
    for (t, &count) in h.iter().enumerate() {
        above -= count;
        if above as f32 <= allowed {
            return t;
        }
    }
    h.len() - 1
}

/// C5. The mean and the variance of the grey level of a random pixel, in bin numbers
/// (slide "The math: the histogram as a distribution"): mean = sum of i p_i,
/// variance = sum of (i - mean)² p_i, with p_i = h_i / N.
pub fn mean_variance(h: &[usize]) -> (f32, f32) {
    let n: usize = h.iter().sum();
    let mut mean = 0.0;
    for (i, &count) in h.iter().enumerate() {
        mean += i as f32 * count as f32 / n as f32;
    }
    let mut var = 0.0;
    for (i, &count) in h.iter().enumerate() {
        var += (i as f32 - mean).powi(2) * count as f32 / n as f32;
    }
    (mean, var)
}
