// Model solution for the coding exercises of "Intensity inside an object".

use imgcourse::Image;

/// The statistics of the pixel values inside one object (slide "Statistics of the pixel values").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub integrated: f32,
    pub mean: f32,
    /// Standard deviation, dividing by `n` (the pixels are the whole object).
    pub sd: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub q1: f32,
    pub q3: f32,
    pub mad: f32,
}

/// C1a. The quantile `p` (between 0 and 1) of a list of values: sort them, take position
/// `h = (n − 1) p` counting from 0, and interpolate linearly between the two neighbours.
pub fn quantile(values: &[f32], p: f32) -> f32 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = (v.len() - 1) as f32 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let t = h - lo as f32;
    v[lo] + t * (v[hi] - v[lo])
}

/// C1b. The median absolute deviation: the median of the distances `|v − median|`.
pub fn mad(values: &[f32]) -> f32 {
    let m = quantile(values, 0.5);
    let dist: Vec<f32> = values.iter().map(|v| (v - m).abs()).collect();
    quantile(&dist, 0.5)
}

/// C1c. The robust z-score of `x` against control values: `(x − median) / (1.4826 · MAD)`.
pub fn robust_z(x: f32, controls: &[f32]) -> f32 {
    (x - quantile(controls, 0.5)) / (1.4826 * mad(controls))
}

/// C2. All statistics of the values of the pixels with `labels[(x, y)] == label`.
pub fn object_stats(img: &Image<f32>, labels: &Image<u32>, label: u32) -> Stats {
    let mut values = Vec::new();
    for (x, y) in img.coords() {
        if labels[(x, y)] == label {
            values.push(img[(x, y)]);
        }
    }
    let n = values.len();
    let integrated: f32 = values.iter().sum();
    let mean = integrated / n as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    Stats {
        n,
        integrated,
        mean,
        sd: var.sqrt(),
        min: values.iter().copied().fold(f32::INFINITY, f32::min),
        max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        median: quantile(&values, 0.5),
        q1: quantile(&values, 0.25),
        q3: quantile(&values, 0.75),
        mad: mad(&values),
    }
}

/// C3. The edge band of an object: its pixels that have a 4-neighbour outside the object
/// (another label, or outside the image). This is the object minus its erosion by the cross.
pub fn edge_band(labels: &Image<u32>, label: u32) -> Image<bool> {
    labels.map_xy(|x, y, l| {
        if l != label {
            return false;
        }
        let (x, y) = (x as isize, y as isize);
        [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
            let (nx, ny) = (x + dx, y + dy);
            !labels.contains(nx, ny) || labels[(nx as usize, ny as usize)] != label
        })
    })
}

/// C4. The geometric centroid, the intensity-weighted centroid, and the mass displacement
/// (the distance between them) of an object. Centroids are `(x, y)`.
pub fn mass_displacement(img: &Image<f32>, labels: &Image<u32>, label: u32) -> ([f32; 2], [f32; 2], f32) {
    let (mut n, mut sx, mut sy) = (0.0, 0.0, 0.0); // m00, m10, m01
    let (mut w, mut wx, mut wy) = (0.0, 0.0, 0.0); // M00, M10, M01
    for (x, y) in img.coords() {
        if labels[(x, y)] == label {
            let v = img[(x, y)];
            n += 1.0;
            sx += x as f32;
            sy += y as f32;
            w += v;
            wx += v * x as f32;
            wy += v * y as f32;
        }
    }
    let geometric = [sx / n, sy / n];
    let weighted = [wx / w, wy / w];
    let d = ((weighted[0] - geometric[0]).powi(2) + (weighted[1] - geometric[1]).powi(2)).sqrt();
    (geometric, weighted, d)
}

/// C5a. The ring (1 to `k`) of a pixel at distance `dc` from the centre and `de` from the
/// edge: `ρ = dc / (dc + de)`, ring `min(k, ⌊k ρ⌋ + 1)`.
pub fn ring_of(dc: f32, de: f32, k: usize) -> usize {
    let rho = dc / (dc + de);
    ((k as f32 * rho).floor() as usize + 1).min(k)
}

/// C5b. FracAtD and MeanFrac of each ring, from the pixel values and their rings (1 to `k`):
/// FracAtD = the ring's share of the total intensity, MeanFrac = FracAtD / the ring's
/// share of the pixels. Returns `(frac_at_d, mean_frac)`, each with `k` entries.
pub fn radial_distribution(values: &[f32], rings: &[usize], k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sum = vec![0.0; k];
    let mut count = vec![0.0; k];
    for (v, &r) in values.iter().zip(rings) {
        sum[r - 1] += v;
        count[r - 1] += 1.0;
    }
    let total: f32 = sum.iter().sum();
    let n = values.len() as f32;
    let frac: Vec<f32> = sum.iter().map(|s| s / total).collect();
    let mean_frac = (0..k).map(|j| if count[j] > 0.0 { frac[j] / (count[j] / n) } else { 0.0 }).collect();
    (frac, mean_frac)
}
