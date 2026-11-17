// Coding exercises for "Lifetimes, decays and photon counting". Replace each `todo!()` with
// your code, then check it with `cargo test -p lifetime` and look at `cargo run -p lifetime`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `rng.uniform()` is in [0, 1), so `-tau * (1.0 - rng.uniform()).ln()` is an exponential
/// waiting time. `t.rem_euclid(period)` wraps a time into [0, period). Time t falls in bin
/// `(t / period * bins as f32) as usize` (keep it below `bins`).
pub fn simulate(tau: f32, photons: usize, background: f32, setup: &Setup, rng: &mut Rng) -> Vec<f32> {
    todo!("C1: simulate")
}

/// C2. The mean arrival time of a histogram: the count-weighted mean of the bin centres.
/// (The lifetime estimate is this minus the IRF's mean arrival time.)
pub fn mean_arrival(counts: &[f32], period: f32) -> f32 {
    todo!("C2: mean_arrival")
}

/// C3. The two-gate estimate: count the photons D1 in the bins whose centres lie in
/// [start, start + width) and D2 in [start + width, start + 2 width); τ = width / ln(D1 / D2).
pub fn two_gates(counts: &[f32], period: f32, start: f32, width: f32) -> f32 {
    todo!("C3: two_gates")
}

/// C4. A lifetime image from a FLIM stack (`stack[(x, y, j)]` = photons of pixel (x, y) in time
/// bin j). For every pixel: sum the histograms of the (2 `radius` + 1)² pixels around it (binning;
/// only those inside the image), and if that sum holds at least `min_photons` photons, the
/// lifetime is its mean arrival time minus `irf_mean`. Pixels with fewer photons are masked: 0.
///
/// Hint: `stack.width()`, `stack.height()` and `stack.depth()` (the number of time bins);
/// `Image::from_fn(w, h, |x, y| ...)` builds the result pixel by pixel; reuse `mean_arrival`.
pub fn lifetime_image(stack: &Volume<f32>, period: f32, irf_mean: f32, radius: usize, min_photons: f32) -> Image<f32> {
    todo!("C4: lifetime_image")
}
