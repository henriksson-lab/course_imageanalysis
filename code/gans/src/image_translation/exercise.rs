// Coding exercises for "Image-to-image translation and hallucination". Replace each `todo!()` with your code,
// then check it with `cargo test -p gans image_translation::` and look at `cargo run -p gans -- image-translation`.
#![allow(unused_variables, unused_imports, unused_mut)]

use imgcourse::Rng;

/// One pixel of a training pair: input x (label-free), target y (stain), and whether the hidden
/// marker was on (known only because the data are simulated).
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub x: f64,
    pub y: f64,
    pub on: bool,
}

/// The clean target without the marker: y = 0.2 + 0.4 x.
pub fn base(x: f64) -> f64 {
    0.2 + 0.4 * x
}

/// n simulated pairs: x uniform in [0, 1]; for x > 0.5 the marker is on with probability `q`
/// and adds `gap`; noise of standard deviation 0.03.
pub fn pairs(n: usize, gap: f64, q: f64, rng: &mut Rng) -> Vec<Pair> {
    (0..n)
        .map(|_| {
            let x = rng.uniform() as f64;
            let on = x > 0.5 && (rng.uniform() as f64) < q;
            let y = base(x) + if on { gap } else { 0.0 } + 0.03 * rng.normal() as f64;
            Pair { x, y, on }
        })
        .collect()
}

/// Kernel weights exp(−½((x_i − x)/h)²) of the training pairs at the input x (given; use it
/// in C1–C3).
pub fn weights(train: &[Pair], x: f64, h: f64) -> Vec<f64> {
    train.iter().map(|p| (-0.5 * ((p.x - x) / h).powi(2)).exp()).collect()
}

/// C1. What a translator trained with the squared error learns: the conditional mean, here
/// the kernel-weighted mean of the targets of the training pairs near x (bandwidth h).
pub fn cond_mean(train: &[Pair], x: f64, h: f64) -> f64 {
    todo!("C1: cond_mean")
}

/// C2. What a translator trained with the absolute error learns: the conditional median, here
/// the weighted median (sort the targets; the first whose cumulative weight reaches half).
pub fn cond_median(train: &[Pair], x: f64, h: f64) -> f64 {
    todo!("C2: cond_median")
}

/// C3. What a perfect conditional GAN does: draw the target of a random training pair, chosen
/// with probability proportional to its weight at x.
pub fn cond_sample(train: &[Pair], x: f64, h: f64, rng: &mut Rng) -> f64 {
    todo!("C3: cond_sample")
}

/// C4. Unpaired translation: map x to the target value with the same rank (quantile) among the
/// training targets, or the opposite rank if `decreasing`. Uses only the two separate lists.
pub fn quantile_map(xs: &[f64], ys: &[f64], x: f64, decreasing: bool) -> f64 {
    todo!("C4: quantile_map")
}

/// C5. Root mean squared error between predictions and the true targets.
pub fn rmse(pred: &[f64], truth: &[f64]) -> f64 {
    todo!("C5: rmse")
}

/// C5. Could a real cell at input x have the target y? Within 3 noise sd (0.09) of the branch
/// without the marker, or (for x > 0.5) of the branch with it.
pub fn is_real(x: f64, y: f64, gap: f64) -> bool {
    todo!("C5: is_real")
}
