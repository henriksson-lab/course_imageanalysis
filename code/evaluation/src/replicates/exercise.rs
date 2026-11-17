// Coding exercises for "From measurements to claims". Replace each `todo!()` with your code,
// then check it with `cargo test -p evaluation replicates::` and look at `cargo run -p evaluation -- replicates`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: the mean first, then the sum of squared differences from it.
pub fn mean_sd(x: &[f32]) -> (f32, f32) {
    todo!("C1: mean_sd")
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
///
/// Hint: make the list of differences and use `mean_sd` on it.
pub fn paired_t(control: &[f32], treated: &[f32]) -> TTest {
    todo!("C2a: paired_t")
}

/// C2b. Unpaired t-test of two groups: estimate = mean(y) - mean(x),
/// SE = sqrt(sd_x^2 / n_x + sd_y^2 / n_y), df = n_x + n_y - 2.
///
/// Hint: `mean_sd` of each group; the estimate is the difference of the two means.
pub fn unpaired_t(x: &[f32], y: &[f32]) -> TTest {
    todo!("C2b: unpaired_t")
}

/// C3. How often the two analyses find an effect that is not there. Simulate `runs` studies
/// with no treatment effect: `experiments` experiments per condition, each with its own
/// day offset (normal, SD `sd_day`), and `cells` cells per experiment (normal around it,
/// SD `sd_cell`). In each study, test (a) all cells as n, (b) the experiment means as n,
/// both with `unpaired_t`. Returns the share of significant results of (a) and of (b).
///
/// Hint: per study, keep two lists per condition: all cells, and one mean per experiment.
/// All cells of one experiment share the same `day = sd_day * rng.normal()`.
/// `unpaired_t(..).significant()` does the test.
pub fn false_positive_rates(experiments: usize, cells: usize, sd_day: f32, sd_cell: f32, runs: usize, seed: u64) -> (f32, f32) {
    todo!("C3: false_positive_rates")
}

/// C4a. Bonferroni: the number of p-values `<= alpha / m`.
pub fn bonferroni(p: &[f32], alpha: f32) -> usize {
    todo!("C4a: bonferroni")
}

/// C4b. Benjamini–Hochberg at level `q`: sort the p-values; find the largest `k` (from 1)
/// with `p_(k) <= k q / m`; the `k` smallest p-values are hits. Returns `k` (0 if none).
///
/// Hint: sort a copy with `sorted.sort_by(|a, b| a.total_cmp(b))`; go through all `k` and
/// remember the largest one that passes (do not stop at the first one that fails).
pub fn benjamini_hochberg(p: &[f32], q: f32) -> usize {
    todo!("C4b: benjamini_hochberg")
}
