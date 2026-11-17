// Model solution for the coding exercises of "Crowded spots and evaluating a spot detector".

/// C1. The number of local maxima of a 1D profile: samples larger than both neighbours (the
/// first and last sample are not counted).
pub fn count_peaks(profile: &[f32]) -> usize {
    let mut n = 0;
    for i in 1..profile.len().saturating_sub(1) {
        if profile[i] > profile[i - 1] && profile[i] > profile[i + 1] {
            n += 1;
        }
    }
    n
}

/// The distance between two points.
pub fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// C2. Greedy matching: of all truth–detection pairs within `gate`, take the closest first,
/// then the next closest whose truth and detection are both still free, and so on.
/// Returns the pairs `(truth index, detection index)`.
pub fn match_greedy(truth: &[[f32; 2]], det: &[[f32; 2]], gate: f32) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    for (i, &t) in truth.iter().enumerate() {
        for (j, &d) in det.iter().enumerate() {
            if dist(t, d) <= gate {
                candidates.push((dist(t, d), i, j));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut t_used, mut d_used) = (vec![false; truth.len()], vec![false; det.len()]);
    let mut pairs = Vec::new();
    for (_, i, j) in candidates {
        if !t_used[i] && !d_used[j] {
            t_used[i] = true;
            d_used[j] = true;
            pairs.push((i, j));
        }
    }
    pairs
}

/// C3. Optimal matching for a small group: of all one-to-one matchings with pairs within `gate`,
/// the one with the **most** pairs, and among those the smallest total distance. Tries every
/// possibility (truth by truth: unmatched, or with any free detection in its gate), so use it
/// only on small groups (`sim::match_in_groups` splits a whole image into such groups).
pub fn match_optimal(truth: &[[f32; 2]], det: &[[f32; 2]], gate: f32) -> Vec<(usize, usize)> {
    // best (count, total distance, pairs) for truths i.. with the detections in `used` taken
    fn search(i: usize, truth: &[[f32; 2]], det: &[[f32; 2]], gate: f32, used: &mut [bool]) -> (usize, f32, Vec<(usize, usize)>) {
        if i == truth.len() {
            return (0, 0.0, Vec::new());
        }
        let mut best = search(i + 1, truth, det, gate, used); // truth i unmatched
        for j in 0..det.len() {
            let d = dist(truth[i], det[j]);
            if !used[j] && d <= gate {
                used[j] = true;
                let (n, total, mut pairs) = search(i + 1, truth, det, gate, used);
                used[j] = false;
                let (n, total) = (n + 1, total + d);
                if n > best.0 || (n == best.0 && total < best.1) {
                    pairs.push((i, j));
                    best = (n, total, pairs);
                }
            }
        }
        best
    }
    let mut pairs = search(0, truth, det, gate, &mut vec![false; det.len()]).2;
    pairs.sort();
    pairs
}

/// The scores of a matching.
#[derive(Clone, Copy, Debug)]
pub struct Scores {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub jaccard: f32,
    /// root mean square distance of the matched pairs
    pub rmse: f32,
}

/// C4. TP, FP, FN, precision, recall, F1, Jaccard and the RMSE of the matched distances, from the
/// pairs of a matching.
pub fn scores(truth: &[[f32; 2]], det: &[[f32; 2]], pairs: &[(usize, usize)]) -> Scores {
    let tp = pairs.len();
    let fp = det.len() - tp;
    let fn_ = truth.len() - tp;
    let ratio = |a: usize, b: usize| if b > 0 { a as f32 / b as f32 } else { 0.0 };
    let sum_sq: f32 = pairs.iter().map(|&(i, j)| dist(truth[i], det[j]).powi(2)).sum();
    Scores {
        tp,
        fp,
        fn_,
        precision: ratio(tp, tp + fp),
        recall: ratio(tp, tp + fn_),
        f1: ratio(2 * tp, 2 * tp + fp + fn_),
        jaccard: ratio(tp, tp + fp + fn_),
        rmse: if tp > 0 { (sum_sq / tp as f32).sqrt() } else { 0.0 },
    }
}

/// C5. The precision–recall curve of detections ranked by score (`is_tp[k]`: is the k-th
/// strongest detection matched?), with `n_truth` true spots. Returns the points
/// `(precision, recall)` after each detection, and the average precision: precision made
/// non-increasing from the right, then summed over the recall steps (precision × step).
pub fn pr_curve(is_tp: &[bool], n_truth: usize) -> (Vec<(f32, f32)>, f32) {
    let mut curve = Vec::new();
    let mut tp = 0;
    for (k, &hit) in is_tp.iter().enumerate() {
        if hit {
            tp += 1;
        }
        curve.push((tp as f32 / (k + 1) as f32, tp as f32 / n_truth as f32));
    }
    // non-increasing precision from the right
    let mut p: Vec<f32> = curve.iter().map(|c| c.0).collect();
    for k in (0..p.len().saturating_sub(1)).rev() {
        p[k] = p[k].max(p[k + 1]);
    }
    let mut ap = 0.0;
    let mut last_recall = 0.0;
    for (k, c) in curve.iter().enumerate() {
        ap += p[k] * (c.1 - last_recall);
        last_recall = c.1;
    }
    (curve, ap)
}
