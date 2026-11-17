// Model solution for the coding exercises of "Samplers, steps, and judging the samples".

/// C1. Deterministic sampling as Euler steps of the probability-flow equation
/// `dy/dσ = (y − D(y, σ)) / σ`: start at `y` with the noise level `sigmas[0]` and step through
/// the levels `sigmas` (falling, the last one may be 0). Each step from σ to σ′ is
/// `y ← y + (σ′ − σ) · (y − D(y, σ)) / σ`, where `denoise(y, σ)` is the best denoiser `D`.
pub fn euler_sample(y: f32, sigmas: &[f32], denoise: impl Fn(f32, f32) -> f32) -> f32 {
    let mut y = y;
    for k in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[k], sigmas[k + 1]);
        let slope = (y - denoise(y, s)) / s;
        y += (next - s) * slope;
    }
    y
}

/// The mean and standard deviation (dividing by n) of a set of values.
pub fn mean_sd(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    (mean, (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt())
}

/// C2. The Fréchet distance of two sets of numbers in 1D: fit a Gaussian to each (mean and
/// standard deviation, dividing by n) and return `(μ1 − μ2)² + (s1 − s2)²`.
pub fn frechet_1d(a: &[f32], b: &[f32]) -> f32 {
    let (ma, sa) = mean_sd(a);
    let (mb, sb) = mean_sd(b);
    (ma - mb).powi(2) + (sa - sb).powi(2)
}

/// The distance from `x` to its `k`-th nearest neighbour in `set` (the point itself, at
/// distance 0, counts as the first if it is in the set).
pub fn kth_distance(x: f32, set: &[f32], k: usize) -> f32 {
    let mut d: Vec<f32> = set.iter().map(|s| (s - x).abs()).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    d[k.min(d.len() - 1)]
}

/// C3. Precision and recall with `k`-NN balls. Around every real point put a ball whose
/// radius is the distance to its `k`-th nearest real neighbour (`kth_distance(r, real, k)`);
/// **precision** is the fraction of generated points inside at least one real ball. In the
/// same way with balls around the generated points, **recall** is the fraction of real points
/// inside at least one generated ball. Returns `(precision, recall)`.
pub fn precision_recall(real: &[f32], generated: &[f32], k: usize) -> (f32, f32) {
    let covered = |points: &[f32], centres: &[f32]| {
        let radius: Vec<f32> = centres.iter().map(|&c| kth_distance(c, centres, k)).collect();
        let inside = points.iter().filter(|&&p| centres.iter().zip(&radius).any(|(&c, &r)| (p - c).abs() <= r)).count();
        inside as f32 / points.len() as f32
    };
    (covered(generated, real), covered(real, generated))
}

/// C4. The memorisation check: for every sample, the distance to its nearest training point.
pub fn nearest_distances(samples: &[f32], training: &[f32]) -> Vec<f32> {
    samples.iter().map(|&s| training.iter().map(|&t| (s - t).abs()).fold(f32::MAX, f32::min)).collect()
}
