// Model solution for the coding exercises of "Choosing the threshold from the histogram".
//
// Everything works on a histogram `h` of counts: `h[i]` pixels have grey level (bin) `i`.
// A threshold `t` splits it into class 0 (levels 0..=t, background) and class 1 (levels
// above t, object).

/// R1. The weight of class 0 and the means of both classes for the split at `t`:
/// `(w0, mu0, mu1)`, with `w0` the fraction of pixels at or below `t`. A class without
/// pixels gets the mean 0.
pub fn class_stats(h: &[usize], t: usize) -> (f64, f64, f64) {
    let (mut n0, mut s0, mut n1, mut s1) = (0.0, 0.0, 0.0, 0.0);
    for (i, &count) in h.iter().enumerate() {
        let c = count as f64;
        if i <= t {
            n0 += c;
            s0 += i as f64 * c;
        } else {
            n1 += c;
            s1 += i as f64 * c;
        }
    }
    let mean = |s: f64, n: f64| if n > 0.0 { s / n } else { 0.0 };
    (n0 / (n0 + n1), mean(s0, n0), mean(s1, n1))
}

/// R2. Otsu's threshold: the `t` with the largest between-class variance
/// `w0 · w1 · (mu0 − mu1)²`. Splits with an empty class are skipped; on a tie the smallest `t` wins.
pub fn otsu(h: &[usize]) -> usize {
    let mut best = (0, -1.0);
    for t in 0..h.len() - 1 {
        let (w0, mu0, mu1) = class_stats(h, t);
        if w0 == 0.0 || w0 == 1.0 {
            continue;
        }
        let between = w0 * (1.0 - w0) * (mu0 - mu1).powi(2);
        if between > best.1 {
            best = (t, between);
        }
    }
    best.0
}

/// R3. Isodata from the start value `t`: split at `t` (class 0 = levels `<= t`), move `t` to
/// the midpoint of the two class means, and repeat until the split no longer changes.
/// Returns the last `t` (a value between levels, like 3.53).
pub fn isodata(h: &[usize], start: f64) -> f64 {
    let mut t = start;
    loop {
        let (_, mu0, mu1) = class_stats(h, t.floor() as usize);
        let next = (mu0 + mu1) / 2.0;
        if next.floor() == t.floor() {
            return next;
        }
        t = next;
    }
}

/// R4. The minimum-error threshold of Kittler and Illingworth: fit a normal distribution
/// to each class (weight, mean, variance) and keep the `t` with the smallest
/// `J(t) = 1 + 2 (w0 ln s0 + w1 ln s1) − 2 (w0 ln w0 + w1 ln w1)` (slide "The math:
/// Kittler–Illingworth"). Splits where a class is empty or has variance 0 are skipped.
pub fn min_error(h: &[usize]) -> usize {
    let total: f64 = h.iter().map(|&c| c as f64).sum();
    let mut best = (0, f64::INFINITY);
    for t in 0..h.len() - 1 {
        // weight, mean and variance of the levels lo..hi
        let fit = |lo: usize, hi: usize| {
            let (mut n, mut s, mut s2) = (0.0, 0.0, 0.0);
            for i in lo..hi {
                let c = h[i] as f64;
                n += c;
                s += i as f64 * c;
                s2 += (i * i) as f64 * c;
            }
            let mean = s / n;
            (n / total, s2 / n - mean * mean)
        };
        let (w0, v0) = fit(0, t + 1);
        let (w1, v1) = fit(t + 1, h.len());
        if w0 == 0.0 || w1 == 0.0 || v0 <= 0.0 || v1 <= 0.0 {
            continue;
        }
        // ln s = ln(variance) / 2
        let j = 1.0 + (w0 * v0.ln() + w1 * v1.ln()) - 2.0 * (w0 * w0.ln() + w1 * w1.ln());
        if j < best.1 {
            best = (t, j);
        }
    }
    best.0
}

/// R5. Otsu with three classes: the pair `t1 < t2` (classes `0..=t1`, `t1+1..=t2`, above
/// `t2`) with the largest `sum of w_k (mu_k − mu)²`. Splits with an empty class are skipped.
pub fn otsu3(h: &[usize]) -> (usize, usize) {
    let levels = h.len();
    // running sums: n[t] pixels and s[t] sum of levels in 0..t, so that a class lo..hi
    // has n[hi] - n[lo] pixels
    let mut n = vec![0.0; levels + 1];
    let mut s = vec![0.0; levels + 1];
    for i in 0..levels {
        n[i + 1] = n[i] + h[i] as f64;
        s[i + 1] = s[i] + (i * h[i]) as f64;
    }
    let (total, mu) = (n[levels], s[levels] / n[levels]);
    let class = |lo: usize, hi: usize| {
        let (w, sum) = (n[hi] - n[lo], s[hi] - s[lo]);
        if w == 0.0 {
            None
        } else {
            Some(w / total * (sum / w - mu).powi(2))
        }
    };
    let mut best = ((0, 1), -1.0);
    for t1 in 0..levels {
        for t2 in t1 + 1..levels - 1 {
            if let (Some(a), Some(b), Some(c)) = (class(0, t1 + 1), class(t1 + 1, t2 + 1), class(t2 + 1, levels)) {
                if a + b + c > best.1 {
                    best = ((t1, t2), a + b + c);
                }
            }
        }
    }
    best.0
}
