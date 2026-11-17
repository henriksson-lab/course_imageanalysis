// Coding exercises for "Choosing the threshold from the histogram". Replace each `todo!()`
// with your code, then check it with `cargo test -p otsu` and look at `cargo run -p otsu`.
#![allow(unused_variables, unused_imports)]
//
// Everything works on a histogram `h` of counts: `h[i]` pixels have grey level (bin) `i`.
// A threshold `t` splits it into class 0 (levels 0..=t, background) and class 1 (levels
// above t, object).

/// R1. The weight of class 0 and the means of both classes for the split at `t`:
/// `(w0, mu0, mu1)`, with `w0` the fraction of pixels at or below `t`. A class without
/// pixels gets the mean 0.
///
/// Hint: one loop over the levels `i`, adding up the counts and `i · h[i]` of each class.
pub fn class_stats(h: &[usize], t: usize) -> (f64, f64, f64) {
    todo!("R1: class_stats")
}

/// R2. Otsu's threshold: the `t` with the largest between-class variance
/// `w0 · w1 · (mu0 − mu1)²`. Splits with an empty class are skipped; on a tie the smallest `t` wins.
///
/// Hint: try every `t` from 0 to `h.len() - 2` with `class_stats`; keep the best so far.
pub fn otsu(h: &[usize]) -> usize {
    todo!("R2: otsu")
}

/// R3. Isodata from the start value `t`: split at `t` (class 0 = levels `<= t`), move `t` to
/// the midpoint of the two class means, and repeat until the split no longer changes.
/// Returns the last `t` (a value between levels, like 3.53).
///
/// Hint: the split of a value `t` is `t.floor() as usize`. Stop when the new `t` has the
/// same floor as the old one.
pub fn isodata(h: &[usize], start: f64) -> f64 {
    todo!("R3: isodata")
}

/// R4. The minimum-error threshold of Kittler and Illingworth: fit a normal distribution
/// to each class (weight, mean, variance) and keep the `t` with the smallest
/// `J(t) = 1 + 2 (w0 ln s0 + w1 ln s1) − 2 (w0 ln w0 + w1 ln w1)` (slide "The math:
/// Kittler–Illingworth"). Splits where a class is empty or has variance 0 are skipped.
///
/// Hint: the variance of a class is (sum of i² h_i) / n − mean², and 2 ln s = ln(variance).
pub fn min_error(h: &[usize]) -> usize {
    todo!("R4: min_error")
}

/// R5. Otsu with three classes: the pair `t1 < t2` (classes `0..=t1`, `t1+1..=t2`, above
/// `t2`) with the largest `sum of w_k (mu_k − mu)²`. Splits with an empty class are skipped.
///
/// Hint: two nested loops over `t1` and `t2`. To keep it fast, first make running sums of
/// the counts and of `i · h[i]`, so that the weight and the mean of any class are two
/// subtractions.
pub fn otsu3(h: &[usize]) -> (usize, usize) {
    todo!("R5: otsu3")
}
