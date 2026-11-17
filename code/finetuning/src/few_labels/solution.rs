// Model solution for the coding exercises of "Few labels: annotation and the human in the loop".

pub use super::model::*;

/// C1. Uncertainty of a two-class prediction p = p(class 1): 1 when p = 0.5, 0 when p is 0 or 1
/// (one minus the margin |2p − 1|).
pub fn uncertainty(p: f32) -> f32 {
    1.0 - (2.0 * p - 1.0).abs()
}

/// C2. The unlabelled cell whose prediction is most uncertain (`None` if all are labelled).
/// Ties go to the lowest index.
pub fn most_uncertain(probs: &[f32], labelled: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for j in 0..probs.len() {
        if labelled[j] {
            continue;
        }
        if best.is_none_or(|b| uncertainty(probs[j]) > uncertainty(probs[b])) {
            best = Some(j);
        }
    }
    best
}

/// C3. The unlabelled cell farthest from its nearest labelled cell (`None` if all are labelled):
/// the most different from everything labelled so far.
pub fn most_different(cells: &[Cell], labelled: &[bool]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for j in 0..cells.len() {
        if labelled[j] {
            continue;
        }
        let mut d = f32::INFINITY;
        for i in 0..cells.len() {
            if labelled[i] {
                d = d.min(((cells[i].x - cells[j].x).powi(2) + (cells[i].y - cells[j].y).powi(2)).sqrt());
            }
        }
        if best.is_none_or(|(_, bd)| d > bd) {
            best = Some((j, d));
        }
    }
    best.map(|(j, _)| j)
}

/// C4. Binary entropy of a prediction in bits: −p log₂ p − (1 − p) log₂ (1 − p) (0 at p = 0 or 1).
pub fn entropy(p: f32) -> f32 {
    let h = |q: f32| if q > 0.0 { -q * q.log2() } else { 0.0 };
    h(p) + h(1.0 - p)
}
