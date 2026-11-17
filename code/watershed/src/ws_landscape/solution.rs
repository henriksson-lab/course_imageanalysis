// Model solution for the coding exercises of "Images as landscapes: watershed by flooding".
// All functions work on a 1D signal (a profile), samples 0, 1, ..., n − 1.

/// The size of the derivative by central differences, |f(i+1) − f(i−1)| / 2, with the end
/// samples repeated (f(−1) = f(0), f(n) = f(n−1)). (A helper, as in exercise B4.)
pub fn abs_derivative(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    (0..n).map(|i| (f[(i + 1).min(n - 1)] - f[i.saturating_sub(1)]).abs() / 2.0).collect()
}

/// C1. The regional minima: runs of equal samples whose neighbours (the sample left of the
/// run and the one right of it, where they exist) are both higher. Returns a label per sample:
/// 1, 2, 3, ... for the minima from left to right, 0 elsewhere.
pub fn regional_minima(f: &[f32]) -> Vec<u32> {
    let n = f.len();
    let mut labels = vec![0; n];
    let mut count = 0;
    let mut start = 0;
    while start < n {
        // the run of equal values start..end
        let mut end = start + 1;
        while end < n && f[end] == f[start] {
            end += 1;
        }
        let left_higher = start == 0 || f[start - 1] > f[start];
        let right_higher = end == n || f[end] > f[start];
        if left_higher && right_higher {
            count += 1;
            for l in &mut labels[start..end] {
                *l = count;
            }
        }
        start = end;
    }
    labels
}

/// C2. A drop of water placed on sample `start`: at each step it moves to the lower of its two
/// neighbours, if that is lower than where it is (on a tie, to the left). Returns the sample
/// where it stops.
pub fn drop_of_water(f: &[f32], start: usize) -> usize {
    let mut i = start;
    loop {
        let mut next = i;
        if i > 0 && f[i - 1] < f[next] {
            next = i - 1;
        }
        if i + 1 < f.len() && f[i + 1] < f[next] {
            next = i + 1;
        }
        if next == i {
            return i;
        }
        i = next;
    }
}

/// C3. Flooding from the labelled `sources` (0 = not a source). For every water level L, the
/// distinct heights in increasing order, grow all lakes at the same speed, in rounds:
/// (a) every unlabelled sample with f ≤ L next to a lake takes its label (the left neighbour's
/// if both are labelled), all samples at once, so that each lake grows by one sample;
/// (b) water runs downhill at once: an unlabelled sample with f < L next to a lake takes its
/// label straight away (sweep from left to right, repeat until nothing changes).
/// Repeat the rounds until nothing changes, then go to the next level. Returns the labels.
pub fn flood(f: &[f32], sources: &[u32]) -> Vec<u32> {
    let n = f.len();
    let mut labels = sources.to_vec();
    let mut levels = f.to_vec();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    levels.dedup();
    for &level in &levels {
        loop {
            // (a) decide from the labels of the last round, so all lakes grow by one sample
            let mut next = labels.clone();
            let mut changed = false;
            for i in 0..n {
                if labels[i] != 0 || f[i] > level {
                    continue;
                }
                let left = if i > 0 { labels[i - 1] } else { 0 };
                let right = if i + 1 < n { labels[i + 1] } else { 0 };
                let l = if left != 0 { left } else { right };
                if l != 0 {
                    next[i] = l;
                    changed = true;
                }
            }
            labels = next;
            // (b) samples below the water level are flooded at once (an unmarked puddle is drowned)
            let mut spilled = true;
            while spilled {
                spilled = false;
                for i in 0..n {
                    if labels[i] != 0 || f[i] >= level {
                        continue;
                    }
                    let left = if i > 0 { labels[i - 1] } else { 0 };
                    let right = if i + 1 < n { labels[i + 1] } else { 0 };
                    let l = if left != 0 { left } else { right };
                    if l != 0 {
                        labels[i] = l;
                        spilled = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    labels
}

/// C4. The dams: wherever two neighbouring samples i, i + 1 have different labels, the dam is
/// on the higher of the two (the left one if they are equal), and its height is that sample's
/// height. Returns (position, height) for every dam, from left to right.
pub fn dams(f: &[f32], labels: &[u32]) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    for i in 0..labels.len().saturating_sub(1) {
        if labels[i] != labels[i + 1] {
            let at = if f[i + 1] > f[i] { i + 1 } else { i };
            out.push((at, f[at]));
        }
    }
    out
}

/// C5. The h-minima transform: fill every basin by h. Start with g = f + h and erode it,
/// never going below f: g(i) = max(f(i), min(g(i−1), g(i), g(i+1))), all samples at once,
/// until nothing changes. Its regional minima are the minima of f deeper than h.
pub fn h_minima(f: &[f32], h: f32) -> Vec<f32> {
    let n = f.len();
    let mut g: Vec<f32> = f.iter().map(|v| v + h).collect();
    loop {
        let next: Vec<f32> = (0..n)
            .map(|i| {
                let lo = g[i.saturating_sub(1)].min(g[i]).min(g[(i + 1).min(n - 1)]);
                lo.max(f[i])
            })
            .collect();
        if next == g {
            return g;
        }
        g = next;
    }
}
