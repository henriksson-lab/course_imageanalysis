// Model solution for the coding exercises of "Three images, two controls".

use imgcourse::Image;

// The images are photon counts with the background already subtracted:
// idd = donor excitation, donor filter; ida = donor excitation, acceptor filter (the FRET
// channel); iaa = acceptor excitation, acceptor filter.

/// C1. The slope through the origin of `y` against `x` over the mask, Σ x·y / Σ x²
/// (slide "The controls"): on a donor-only sample with x = I_DD, y = I_DA it is the
/// bleed-through d; on an acceptor-only sample with x = I_AA, y = I_DA the cross-excitation a.
pub fn slope_through_origin(x: &Image<f32>, y: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut sxy, mut sxx) = (0.0f64, 0.0f64);
    for p in x.coords().filter(|&p| mask[p]) {
        sxy += (x[p] * y[p]) as f64;
        sxx += (x[p] * x[p]) as f64;
    }
    (sxy / sxx) as f32
}

/// C2a. The corrected sensitised emission at every pixel (slide "Two contaminations"):
/// F_c = I_DA − d · I_DD − a · I_AA.
pub fn corrected_fret(idd: &Image<f32>, ida: &Image<f32>, iaa: &Image<f32>, d: f32, a: f32) -> Image<f32> {
    Image::from_fn(idd.width(), idd.height(), |x, y| ida[(x, y)] - d * idd[(x, y)] - a * iaa[(x, y)])
}

/// C2b. The standard deviation of F_c from photon noise (slide "Subtracting large numbers:
/// noise"): sqrt(I_DA + d² I_DD + a² I_AA), for one pixel (or one sum over pixels).
pub fn fc_noise(idd: f32, ida: f32, iaa: f32, d: f32, a: f32) -> f32 {
    (ida + d * d * idd + a * a * iaa).sqrt()
}

/// C3. The apparent efficiency at every pixel (slide "From F_c to a number we can compare"):
/// E_app = F_c / (F_c + G · I_DD); 0 where the denominator is not positive.
pub fn e_app(fc: &Image<f32>, idd: &Image<f32>, g: f32) -> Image<f32> {
    fc.zip_map(idd, |f, dd| {
        let den = f + g * dd;
        if den > 0.0 {
            f / den
        } else {
            0.0
        }
    })
}

/// The values of one cell: the apparent efficiency and the two indices, from the sums over
/// the cell's pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellValues {
    /// Σ F_c / (Σ F_c + G Σ I_DD)
    pub e_app: f32,
    /// Σ F_c / Σ I_AA
    pub per_acceptor: f32,
    /// Σ F_c / sqrt(Σ I_DD · Σ I_AA)
    pub symmetric: f32,
    /// the number of pixels
    pub area: usize,
}

/// C4. The values of every cell (slide "From F_c to a number we can compare"), from sums
/// over the cell (ratio of sums, not mean of pixel ratios). `labels`: 0 background, 1, 2, …
/// the cells. Returns one entry per label 0, 1, …, max label; entry 0 and labels that do not
/// occur stay at the default (all 0).
pub fn per_cell(fc: &Image<f32>, idd: &Image<f32>, iaa: &Image<f32>, labels: &Image<u32>, g: f32) -> Vec<CellValues> {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut sum_fc = vec![0.0f64; n + 1];
    let mut sum_dd = vec![0.0f64; n + 1];
    let mut sum_aa = vec![0.0f64; n + 1];
    let mut area = vec![0usize; n + 1];
    for p in labels.coords() {
        let l = labels[p] as usize;
        if l > 0 {
            sum_fc[l] += fc[p] as f64;
            sum_dd[l] += idd[p] as f64;
            sum_aa[l] += iaa[p] as f64;
            area[l] += 1;
        }
    }
    let mut out = vec![CellValues::default(); n + 1];
    for l in 1..=n {
        if area[l] == 0 {
            continue;
        }
        let (f, dd, aa) = (sum_fc[l], sum_dd[l], sum_aa[l]);
        out[l] = CellValues {
            e_app: (f / (f + g as f64 * dd)) as f32,
            per_acceptor: (f / aa) as f32,
            symmetric: (f / (dd * aa).sqrt()) as f32,
            area: area[l],
        };
    }
    out
}
