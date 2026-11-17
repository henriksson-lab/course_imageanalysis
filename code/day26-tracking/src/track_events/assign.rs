// Provided, not an exercise: global assignment with alternative costs, as in the first unit
// of the day (track-linking). The solver is the Hungarian algorithm (Kuhn and Munkres, with
// potentials); the lecture only says what it computes.

/// The minimum-cost perfect assignment of a square cost matrix: `result[i]` is the column
/// given to row `i`. Infinite costs (forbidden pairs) are only used if there is no other way.
pub fn hungarian(cost: &[Vec<f32>]) -> Vec<usize> {
    let n = cost.len();
    let big = 1e12; // stands in for infinity
    let c = |i: usize, j: usize| if cost[i][j].is_finite() { cost[i][j] as f64 } else { big };
    // Rows and columns are counted from 1 here; column 0 is a helper.
    let (mut u, mut v) = (vec![0.0f64; n + 1], vec![0.0f64; n + 1]);
    let (mut row_of, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        row_of[0] = i;
        let mut j0 = 0;
        let mut min_v = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = row_of[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let reduced = c(i0 - 1, j - 1) - u[i0] - v[j];
                    if reduced < min_v[j] {
                        min_v[j] = reduced;
                        way[j] = j0;
                    }
                    if min_v[j] < delta {
                        delta = min_v[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[row_of[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }
            j0 = j1;
            if row_of[j0] == 0 {
                break;
            }
        }
        while j0 != 0 {
            let j1 = way[j0];
            row_of[j0] = row_of[j1];
            j0 = j1;
        }
    }
    let mut result = vec![0; n];
    for j in 1..=n {
        if row_of[j] > 0 {
            result[row_of[j] - 1] = j - 1;
        }
    }
    result
}

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
