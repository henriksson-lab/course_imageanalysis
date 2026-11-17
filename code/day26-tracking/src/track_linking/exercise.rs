// Coding exercises for "Tracking: from detections to tracks". Replace each `todo!()` with
// your code, then check it with `cargo test -p day26 track_linking::` and look at `cargo run -p day26 -- track-linking`.
#![allow(unused_variables, unused_imports)]

/// A detection: its position `[x, y]`.
pub type Point = [f32; 2];

/// C1. The cost matrix of all candidate links: `cost[i][j]` is the squared distance from
/// detection `i` at frame t to detection `j` at frame t+1, or infinity (`f32::INFINITY`)
/// if they are farther apart than `gate` (slide "Gating").
///
/// Hint: `vec![vec![f32::INFINITY; to.len()]; from.len()]` is a matrix full of infinity.
pub fn cost_matrix(from: &[Point], to: &[Point], gate: f32) -> Vec<Vec<f32>> {
    todo!("C1: cost_matrix")
}

/// C2. Nearest neighbour: every row takes its cheapest column (`None` if all its costs are
/// infinite). Two rows may take the same column: a conflict.
pub fn nearest(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    todo!("C2: nearest")
}

/// C3. Greedy linking: take the cheapest pair whose row and column are both still free, link
/// it, and repeat until no finite pair is left. Never revisits a choice.
///
/// Hint: keep `links: Vec<Option<usize>>` (a row is free while it is `None`) and
/// `col_used: Vec<bool>`. Each round, search all free pairs for the cheapest finite one.
pub fn greedy(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    todo!("C3: greedy")
}

/// C4. Global assignment by trying all possibilities: give every row a different column so
/// that the total cost is as small as possible (every cost finite). Returns the column of
/// every row, or `None` if no such assignment exists. Needs rows ≤ columns.
///
/// Hint: write a helper function that is called with the columns chosen for the rows
/// before row `i`; it tries every free column with a finite cost for row `i`, and calls
/// itself for row `i + 1`. When all rows have a column, compare the total with the best
/// so far.
pub fn global_by_search(cost: &[Vec<f32>]) -> Option<Vec<usize>> {
    todo!("C4: global_by_search")
}

/// C5a. The augmented matrix with the cost `b` of ending and of starting a track (math slide
/// "Linking as a linear assignment problem"): for m rows and n columns of `cost`, the
/// (m + n) × (n + m) matrix
///
/// ```text
/// [ cost       end   ]    end:   b on the diagonal, ∞ elsewhere (m × m)
/// [ start      0     ]    start: b on the diagonal, ∞ elsewhere (n × n)
/// ```
pub fn augmented(cost: &[Vec<f32>], b: f32) -> Vec<Vec<f32>> {
    todo!("C5a: augmented")
}

/// C5b. Linking with ends and starts: build the augmented matrix, solve it with `solver`
/// (`global_by_search` gives `Option`, so the program passes `hungarian`), and read the links
/// off the top-left block. Row `i` gets `Some(j)` if it links to column `j`, `None` if its
/// track ends.
///
/// Hint: `solver(&augmented(cost, b))` gives a column for every row of the big matrix;
/// for row `i < m`, a column `< n` is a link, a column `≥ n` means "ends".
pub fn link_with_ends(cost: &[Vec<f32>], b: f32, solver: fn(&[Vec<f32>]) -> Vec<usize>) -> Vec<Option<usize>> {
    todo!("C5b: link_with_ends")
}
