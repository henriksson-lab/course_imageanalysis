//! Helpers shared by the units of the day.

/// The mean and standard deviation (dividing by n) of a set of values.
pub fn mean_sd(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    (mean, (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt())
}
