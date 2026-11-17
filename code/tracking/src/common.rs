//! Helpers shared by the units of day 26 (provided, not an exercise): a fast exact solver for the
//! assignment problem (the Hungarian algorithm, in the compact form of Kuhn and Munkres with
//! potentials). The lecture only says what it computes; `global_by_search` (C4 of track-linking)
//! computes the same, slowly.

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
