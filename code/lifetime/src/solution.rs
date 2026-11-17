// Model solution for the coding exercises of "Lifetimes, decays and photon counting".

use imgcourse::{Image, Rng, Volume};

/// The settings of a photon-counting (TCSPC) measurement. Times are in nanoseconds.
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    /// Time between two laser pulses (12.5 ns at 80 MHz). The histogram covers one period.
    pub period: f32,
    /// Number of time bins of the histogram.
    pub bins: usize,
    /// The instrument response (IRF): a Gaussian of width `irf_sigma` centred at `irf_delay`.
    pub irf_delay: f32,
    pub irf_sigma: f32,
}

/// 80 MHz, 64 bins, IRF σ = 0.1 ns at 0.5 ns: the settings of the lifetime-image demo.
pub const DEMO: Setup = Setup { period: 12.5, bins: 64, irf_delay: 0.5, irf_sigma: 0.1 };

/// The centre of time bin `j` of a histogram of `bins` bins over one `period`.
pub fn bin_centre(j: usize, bins: usize, period: f32) -> f32 {
    (j as f32 + 0.5) * period / bins as f32
}

/// C1. Simulate a TCSPC histogram of `photons` photons, photon by photon. A fraction
/// `background` of them are background, at a uniformly random time in the period. Each other
/// photon waits an exponential time of mean `tau` after the pulse, plus the timing jitter of the
/// IRF (delay + σ · a normal random number). Times after the end of the period wrap round into
/// the next one (the slide "Two side effects of the pulse train").
pub fn simulate(tau: f32, photons: usize, background: f32, setup: &Setup, rng: &mut Rng) -> Vec<f32> {
    let mut counts = vec![0.0; setup.bins];
    for _ in 0..photons {
        let t = if rng.uniform() < background {
            setup.period * rng.uniform()
        } else {
            // an exponential waiting time: -τ ln(u) with u uniform in (0, 1]
            let wait = -tau * (1.0 - rng.uniform()).ln();
            setup.irf_delay + setup.irf_sigma * rng.normal() + wait
        };
        let t = t.rem_euclid(setup.period); // wrap-around
        let j = ((t / setup.period * setup.bins as f32) as usize).min(setup.bins - 1);
        counts[j] += 1.0;
    }
    counts
}

/// C2. The mean arrival time of a histogram: the count-weighted mean of the bin centres.
/// (The lifetime estimate is this minus the IRF's mean arrival time.)
pub fn mean_arrival(counts: &[f32], period: f32) -> f32 {
    let (mut sum, mut total) = (0.0, 0.0);
    for (j, &y) in counts.iter().enumerate() {
        sum += y * bin_centre(j, counts.len(), period);
        total += y;
    }
    sum / total
}

/// C3. The two-gate estimate: count the photons D1 in the bins whose centres lie in
/// [start, start + width) and D2 in [start + width, start + 2 width); τ = width / ln(D1 / D2).
pub fn two_gates(counts: &[f32], period: f32, start: f32, width: f32) -> f32 {
    let (mut d1, mut d2) = (0.0, 0.0);
    for (j, &y) in counts.iter().enumerate() {
        let t = bin_centre(j, counts.len(), period);
        if t >= start && t < start + width {
            d1 += y;
        } else if t >= start + width && t < start + 2.0 * width {
            d2 += y;
        }
    }
    width / (d1 / d2).ln()
}

/// C4. A lifetime image from a FLIM stack (`stack[(x, y, j)]` = photons of pixel (x, y) in time
/// bin j). For every pixel: sum the histograms of the (2 `radius` + 1)² pixels around it (binning;
/// only those inside the image), and if that sum holds at least `min_photons` photons, the
/// lifetime is its mean arrival time minus `irf_mean`. Pixels with fewer photons are masked: 0.
pub fn lifetime_image(stack: &Volume<f32>, period: f32, irf_mean: f32, radius: usize, min_photons: f32) -> Image<f32> {
    let (w, h, bins) = (stack.width(), stack.height(), stack.depth());
    let r = radius as isize;
    Image::from_fn(w, h, |x, y| {
        let mut hist = vec![0.0; bins];
        for dy in -r..=r {
            for dx in -r..=r {
                let (xx, yy) = (x as isize + dx, y as isize + dy);
                if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                    for (j, v) in hist.iter_mut().enumerate() {
                        *v += stack[(xx as usize, yy as usize, j)];
                    }
                }
            }
        }
        if hist.iter().sum::<f32>() < min_photons {
            0.0
        } else {
            mean_arrival(&hist, period) - irf_mean
        }
    })
}
