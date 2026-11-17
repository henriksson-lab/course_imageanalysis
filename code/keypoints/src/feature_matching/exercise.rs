// Coding exercises for "Matching descriptors". Replace each `todo!()` with your code,
// then check it with `cargo test -p keypoints feature_matching::` and look at `cargo run -p keypoints -- feature-matching`.
#![allow(unused_variables, unused_imports)]

/// C1. The table of distances: entry `[i][j]` is the Euclidean (L2) distance between
/// descriptor `i` of image 1 and descriptor `j` of image 2.
///
/// Hint: `vec![vec![0.0; d2.len()]; d1.len()]` is a table of zeros.
pub fn distance_table(d1: &[Vec<f32>], d2: &[Vec<f32>]) -> Vec<Vec<f32>> {
    todo!("C1: distance_table")
}

/// C2. The nearest and the second-nearest entry of one row of the table: returns
/// `(j, d1, d2)`, the column of the smallest distance, the smallest and the second smallest
/// distance. The row has at least two entries.
///
/// Hint: start with `d1 = d2 = f32::INFINITY`. A smaller distance than d1 pushes d1 down to d2;
/// one between d1 and d2 replaces d2 only.
pub fn nearest_two(row: &[f32]) -> (usize, f32, f32) {
    todo!("C2: nearest_two")
}

/// C3. The ratio test: for every row `i`, its nearest column `j` is a match if
/// `d1 / d2 < t`. Returns the accepted pairs `(i, j)` in row order.
/// (`t = f32::INFINITY` accepts every nearest neighbour.)
pub fn ratio_test(table: &[Vec<f32>], t: f32) -> Vec<(usize, usize)> {
    todo!("C3: ratio_test")
}

/// C4. The mutual check: keep a match `(i, j)` only if row `i` is also the nearest row of
/// column `j` (the smallest entry of column `j`).
///
/// Hint: for the column `j` of a match, find the row `r` with the smallest `table[r][j]`.
pub fn mutual_check(table: &[Vec<f32>], matches: &[(usize, usize)]) -> Vec<(usize, usize)> {
    todo!("C4: mutual_check")
}

/// C5. Precision and recall of the matches. `truth[i]` is the true partner of point `i` of
/// image 1 (or `None` if it has none). Precision = correct / accepted; recall = correct /
/// possible, where possible = the number of points with a partner. Returns
/// `(correct, precision, recall)`; precision is 0 if nothing was accepted.
///
/// Hint: `truth[i] == Some(j)` tests one match; `truth.iter().filter(|t| t.is_some()).count()`
/// counts the possible ones.
pub fn precision_recall(matches: &[(usize, usize)], truth: &[Option<usize>]) -> (usize, f32, f32) {
    todo!("C5: precision_recall")
}
