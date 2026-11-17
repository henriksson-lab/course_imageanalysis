// Provided, not an exercise: global assignment with alternative costs, as in the first unit
// of the day (track-linking). The solver is the Hungarian algorithm in `src/common.rs`.

use crate::common::hungarian;

/// Global assignment where every row and every column may also stay unjoined, at cost `b`
/// each (the augmented matrix of the first unit). `result[i]` is the column joined to row
/// `i`, or `None`.
pub fn join_with_alternatives(cost: &[Vec<f32>], b: f32) -> Vec<Option<usize>> {
    let m = cost.len();
    let n = if m > 0 { cost[0].len() } else { 0 };
    let mut big = vec![vec![f32::INFINITY; n + m]; m + n];
    for i in 0..m {
        big[i][..n].copy_from_slice(&cost[i]);
        big[i][n + i] = b;
    }
    for j in 0..n {
        big[m + j][j] = b;
        for i in 0..m {
            big[m + j][n + i] = 0.0;
        }
    }
    let assignment = hungarian(&big);
    (0..m).map(|i| if assignment[i] < n { Some(assignment[i]) } else { None }).collect()
}
