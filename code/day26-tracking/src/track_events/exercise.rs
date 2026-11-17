// Coding exercises for "Tracking: gaps, divisions and lineage trees". Replace each `todo!()`
// with your code, then check it with `cargo test -p day26 track_events::` and look at `cargo run -p day26 -- track-events`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `start.t - end.t` is g (check first that it cannot go below 0). Compare squared
/// distances: d² ≤ gate² · g.
pub fn gap_cost(end: &Detection, start: &Detection, max_gap: usize, gate: f32) -> Option<f32> {
    todo!("C1: gap_cost")
}

/// C2. The cost of a split: the `mother` (one frame earlier) continues into `sister`, and a
/// new segment starts at `daughter`. The squared distance from mother to daughter times the
/// area factor max(ρ, 1/ρ)², with ρ = mother area / (daughter area + sister area)
/// (math slide "The segment-linking matrix").
///
/// Hint: `a.max(b)` is the larger of two numbers.
pub fn split_cost(mother: &Detection, daughter: &Detection, sister: &Detection) -> f32 {
    todo!("C2: split_cost")
}

/// C3. Link scores: links are pairs `(from, to)` of detection numbers. Returns precision,
/// recall and F1 of the `result` against the `reference`.
///
/// Hint: a result link is a true positive if `reference.contains(&link)`. Then
/// FP = result links − TP, FN = reference links − TP.
pub fn link_scores(reference: &[(usize, usize)], result: &[(usize, usize)]) -> (f32, f32, f32) {
    todo!("C3: link_scores")
}

/// C4. Division scores: a division is `[mother, daughter, daughter]` (the two daughters in
/// either order). A found division is a true positive only if the mother **and both**
/// daughters match a reference division. Returns (TP, FP, FN).
///
/// Hint: write a small closure `same(a, b)` that compares two divisions, trying both orders
/// of the daughters.
pub fn division_scores(reference: &[[usize; 3]], result: &[[usize; 3]]) -> (usize, usize, usize) {
    todo!("C4: division_scores")
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
///
/// Hint: `if let Some(born) = c.born { match c.divided { ... } }`.
pub fn cycle_times(cells: &[Cell], movie_end: f32) -> (Vec<f32>, Vec<f32>) {
    todo!("C5: cycle_times")
}
