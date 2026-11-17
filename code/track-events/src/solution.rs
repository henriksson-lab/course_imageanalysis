// Model solution for the coding exercises of "Tracking: gaps, divisions and lineage trees".

/// A detection: frame `t`, position `(x, y)` and area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub t: usize,
    pub x: f32,
    pub y: f32,
    pub area: f32,
}

/// The squared distance between two detections (provided).
pub fn squared_distance(a: &Detection, b: &Detection) -> f32 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2)
}

/// C1. The cost of closing a gap from the end of one segment to the start of another
/// (slide "Gap closing"). With g = frames between them: allowed only if 2 ≤ g ≤ `max_gap` + 1
/// and the distance is at most `gate` · √g (the gate grows as √g for random motion). The
/// cost is the squared distance per frame elapsed, d² / g. `None` if not allowed.
pub fn gap_cost(end: &Detection, start: &Detection, max_gap: usize, gate: f32) -> Option<f32> {
    if start.t < end.t + 2 || start.t > end.t + max_gap + 1 {
        return None;
    }
    let g = (start.t - end.t) as f32;
    let d2 = squared_distance(end, start);
    if d2 > gate * gate * g {
        return None;
    }
    Some(d2 / g)
}

/// C2. The cost of a split: the `mother` (one frame earlier) continues into `sister`, and a
/// new segment starts at `daughter`. The squared distance from mother to daughter times the
/// area factor max(ρ, 1/ρ)², with ρ = mother area / (daughter area + sister area)
/// (math slide "The segment-linking matrix").
pub fn split_cost(mother: &Detection, daughter: &Detection, sister: &Detection) -> f32 {
    let rho = mother.area / (daughter.area + sister.area);
    let factor = rho.max(1.0 / rho).powi(2);
    squared_distance(mother, daughter) * factor
}

/// C3. Link scores: links are pairs `(from, to)` of detection numbers. Returns precision,
/// recall and F1 of the `result` against the `reference`.
pub fn link_scores(reference: &[(usize, usize)], result: &[(usize, usize)]) -> (f32, f32, f32) {
    let tp = result.iter().filter(|link| reference.contains(link)).count() as f32;
    let fp = result.len() as f32 - tp;
    let fn_ = reference.len() as f32 - tp;
    (tp / (tp + fp), tp / (tp + fn_), 2.0 * tp / (2.0 * tp + fp + fn_))
}

/// C4. Division scores: a division is `[mother, daughter, daughter]` (the two daughters in
/// either order). A found division is a true positive only if the mother **and both**
/// daughters match a reference division. Returns (TP, FP, FN).
pub fn division_scores(reference: &[[usize; 3]], result: &[[usize; 3]]) -> (usize, usize, usize) {
    let same = |a: &[usize; 3], b: &[usize; 3]| {
        a[0] == b[0] && ((a[1] == b[1] && a[2] == b[2]) || (a[1] == b[2] && a[2] == b[1]))
    };
    let tp = result.iter().filter(|d| reference.iter().any(|r| same(d, r))).count();
    (tp, result.len() - tp, reference.len() - tp)
}

/// A cell of a lineage tree: when it was born (its mother divided), and when it divided
/// itself; `None` if that happened before the movie started or after it ended.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub born: Option<f32>,
    pub divided: Option<f32>,
}

/// C5. Cycle times from a movie that ends at `movie_end`. Only cells born during the movie
/// count. Returns the complete cycles (divided − born), and the observed times of the
/// censored cells (movie_end − born: they had not divided yet when the movie ended).
pub fn cycle_times(cells: &[Cell], movie_end: f32) -> (Vec<f32>, Vec<f32>) {
    let mut complete = Vec::new();
    let mut censored = Vec::new();
    for c in cells {
        if let Some(born) = c.born {
            match c.divided {
                Some(divided) => complete.push(divided - born),
                None => censored.push(movie_end - born),
            }
        }
    }
    (complete, censored)
}
