// Model solution for the coding exercises of "Image-to-image translation and hallucination".

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
    let w = weights(train, x, h);
    let sw: f64 = w.iter().sum();
    train.iter().zip(&w).map(|(p, wi)| wi * p.y).sum::<f64>() / sw
}

/// C2. What a translator trained with the absolute error learns: the conditional median, here
/// the weighted median (sort the targets; the first whose cumulative weight reaches half).
pub fn cond_median(train: &[Pair], x: f64, h: f64) -> f64 {
    let w = weights(train, x, h);
    let mut idx: Vec<usize> = (0..train.len()).collect();
    idx.sort_by(|&a, &b| train[a].y.total_cmp(&train[b].y));
    let half = w.iter().sum::<f64>() / 2.0;
    let mut acc = 0.0;
    for &i in &idx {
        acc += w[i];
        if acc >= half {
            return train[i].y;
        }
    }
    train[idx[idx.len() - 1]].y
}

/// C3. What a perfect conditional GAN does: draw the target of a random training pair, chosen
/// with probability proportional to its weight at x.
pub fn cond_sample(train: &[Pair], x: f64, h: f64, rng: &mut Rng) -> f64 {
    let w = weights(train, x, h);
    let mut u = rng.uniform() as f64 * w.iter().sum::<f64>();
    for (p, wi) in train.iter().zip(&w) {
        u -= wi;
        if u <= 0.0 {
            return p.y;
        }
    }
    train[train.len() - 1].y
}

/// C4. Unpaired translation: map x to the target value with the same rank (quantile) among the
/// training targets, or the opposite rank if `decreasing`. Uses only the two separate lists.
pub fn quantile_map(xs: &[f64], ys: &[f64], x: f64, decreasing: bool) -> f64 {
    let below = xs.iter().filter(|&&v| v < x).count();
    let u = below as f64 / xs.len() as f64;
    let u = if decreasing { 1.0 - u } else { u };
    let mut sy = ys.to_vec();
    sy.sort_by(|a, b| a.total_cmp(b));
    let j = ((u * sy.len() as f64) as usize).min(sy.len() - 1);
    sy[j]
}

/// C5. Root mean squared error between predictions and the true targets.
pub fn rmse(pred: &[f64], truth: &[f64]) -> f64 {
    (pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64).sqrt()
}

/// C5. Could a real cell at input x have the target y? Within 3 noise sd (0.09) of the branch
/// without the marker, or (for x > 0.5) of the branch with it.
pub fn is_real(x: f64, y: f64, gap: f64) -> bool {
    (y - base(x)).abs() <= 0.09 || (x > 0.5 && (y - base(x) - gap).abs() <= 0.09)
}
