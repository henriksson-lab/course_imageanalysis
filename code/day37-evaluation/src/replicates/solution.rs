// Model solution for the coding exercises of "From measurements to claims".

use imgcourse::Rng;

/// The critical value `t_0.975` of the t-distribution with `df` degrees of freedom: a 95 %
/// interval is estimate ± t · SE. (A table; close to 1.96 for many degrees of freedom.)
pub fn t975(df: usize) -> f32 {
    const TABLE: [f32; 30] = [
        12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13, 2.12, 2.11, 2.10, 2.09, 2.09, 2.08, 2.07,
        2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.05, 2.04,
    ];
    if (1..=30).contains(&df) { TABLE[df - 1] } else { 1.96 + 2.4 / df as f32 }
}

/// C1. The mean and the sample standard deviation (divide by n - 1).
pub fn mean_sd(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let ss: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    (mean, (ss / (n - 1.0)).sqrt())
}

/// The result of a t-test: the estimated effect, its standard error, `t = estimate / se`
/// and the degrees of freedom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTest {
    pub estimate: f32,
    pub se: f32,
    pub t: f32,
    pub df: usize,
}

impl TTest {
    /// Significant at 5 % (two-sided): |t| above the critical value.
    pub fn significant(&self) -> bool {
        self.t.abs() > t975(self.df)
    }

    /// The 95 % confidence interval of the effect.
    pub fn interval(&self) -> (f32, f32) {
        (self.estimate - t975(self.df) * self.se, self.estimate + t975(self.df) * self.se)
    }
}

/// C2a. Paired t-test: the differences `treated[i] - control[i]` (one pair per experiment);
/// estimate = their mean, SE = their SD / sqrt(n), df = n - 1.
pub fn paired_t(control: &[f32], treated: &[f32]) -> TTest {
    let d: Vec<f32> = control.iter().zip(treated).map(|(c, t)| t - c).collect();
    let (mean, sd) = mean_sd(&d);
    let se = sd / (d.len() as f32).sqrt();
    TTest { estimate: mean, se, t: mean / se, df: d.len() - 1 }
}

/// C2b. Unpaired t-test of two groups: estimate = mean(y) - mean(x),
/// SE = sqrt(sd_x^2 / n_x + sd_y^2 / n_y), df = n_x + n_y - 2.
pub fn unpaired_t(x: &[f32], y: &[f32]) -> TTest {
    let (mx, sx) = mean_sd(x);
    let (my, sy) = mean_sd(y);
    let se = (sx * sx / x.len() as f32 + sy * sy / y.len() as f32).sqrt();
    TTest { estimate: my - mx, se, t: (my - mx) / se, df: x.len() + y.len() - 2 }
}

/// C3. How often the two analyses find an effect that is not there. Simulate `runs` studies
/// with no treatment effect: `experiments` experiments per condition, each with its own
/// day offset (normal, SD `sd_day`), and `cells` cells per experiment (normal around it,
/// SD `sd_cell`). In each study, test (a) all cells as n, (b) the experiment means as n,
/// both with `unpaired_t`. Returns the share of significant results of (a) and of (b).
pub fn false_positive_rates(experiments: usize, cells: usize, sd_day: f32, sd_cell: f32, runs: usize, seed: u64) -> (f32, f32) {
    let mut rng = Rng::new(seed);
    let (mut cells_hits, mut means_hits) = (0, 0);
    for _ in 0..runs {
        let mut all = [Vec::new(), Vec::new()]; // all cells of control, of treated
        let mut means = [Vec::new(), Vec::new()]; // one mean per experiment
        for condition in 0..2 {
            for _ in 0..experiments {
                let day = sd_day * rng.normal();
                let mut sum = 0.0;
                for _ in 0..cells {
                    let v = day + sd_cell * rng.normal(); // no treatment effect
                    all[condition].push(v);
                    sum += v;
                }
                means[condition].push(sum / cells as f32);
            }
        }
        if unpaired_t(&all[0], &all[1]).significant() {
            cells_hits += 1;
        }
        if unpaired_t(&means[0], &means[1]).significant() {
            means_hits += 1;
        }
    }
    (cells_hits as f32 / runs as f32, means_hits as f32 / runs as f32)
}

/// C4a. Bonferroni: the number of p-values `<= alpha / m`.
pub fn bonferroni(p: &[f32], alpha: f32) -> usize {
    let m = p.len() as f32;
    p.iter().filter(|&&v| v <= alpha / m).count()
}

/// C4b. Benjamini–Hochberg at level `q`: sort the p-values; find the largest `k` (from 1)
/// with `p_(k) <= k q / m`; the `k` smallest p-values are hits. Returns `k` (0 if none).
pub fn benjamini_hochberg(p: &[f32], q: f32) -> usize {
    let mut sorted = p.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let m = sorted.len() as f32;
    let mut hits = 0;
    for (i, &v) in sorted.iter().enumerate() {
        let k = i + 1;
        if v <= k as f32 * q / m {
            hits = k; // keep the largest k, even if smaller k failed
        }
    }
    hits
}
