// Model solution for the coding exercises of "Tracking: from detections to tracks".

/// A detection: its position `[x, y]`.
pub type Point = [f32; 2];

/// C1. The cost matrix of all candidate links: `cost[i][j]` is the squared distance from
/// detection `i` at frame t to detection `j` at frame t+1, or infinity (`f32::INFINITY`)
/// if they are farther apart than `gate` (slide "Gating").
pub fn cost_matrix(from: &[Point], to: &[Point], gate: f32) -> Vec<Vec<f32>> {
    let mut cost = vec![vec![f32::INFINITY; to.len()]; from.len()];
    for (i, a) in from.iter().enumerate() {
        for (j, b) in to.iter().enumerate() {
            let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
            if d2 <= gate * gate {
                cost[i][j] = d2;
            }
        }
    }
    cost
}

/// C2. Nearest neighbour: every row takes its cheapest column (`None` if all its costs are
/// infinite). Two rows may take the same column: a conflict.
pub fn nearest(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let mut links = Vec::new();
    for row in cost {
        let mut best = None;
        let mut best_cost = f32::INFINITY;
        for (j, &c) in row.iter().enumerate() {
            if c < best_cost {
                best_cost = c;
                best = Some(j);
            }
        }
        links.push(best);
    }
    links
}

/// C3. Greedy linking: take the cheapest pair whose row and column are both still free, link
/// it, and repeat until no finite pair is left. Never revisits a choice.
pub fn greedy(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let rows = cost.len();
    let cols = if rows > 0 { cost[0].len() } else { 0 };
    let mut links = vec![None; rows];
    let mut col_used = vec![false; cols];
    loop {
        // the cheapest free pair
        let mut best = None;
        let mut best_cost = f32::INFINITY;
        for i in 0..rows {
            if links[i].is_some() {
                continue;
            }
            for j in 0..cols {
                if !col_used[j] && cost[i][j] < best_cost {
                    best_cost = cost[i][j];
                    best = Some((i, j));
                }
            }
        }
        match best {
            Some((i, j)) => {
                links[i] = Some(j);
                col_used[j] = true;
            }
            None => return links,
        }
    }
}

/// C4. Global assignment by trying all possibilities: give every row a different column so
/// that the total cost is as small as possible (every cost finite). Returns the column of
/// every row, or `None` if no such assignment exists. Needs rows ≤ columns.
pub fn global_by_search(cost: &[Vec<f32>]) -> Option<Vec<usize>> {
    let cols = if cost.is_empty() { 0 } else { cost[0].len() };
    let mut current = Vec::new();
    let mut used = vec![false; cols];
    let mut best = None;
    let mut best_total = f32::INFINITY;
    search(cost, &mut current, &mut used, 0.0, &mut best, &mut best_total);
    best
}

/// Row `current.len()` tries every free column in turn, then the next row does the same.
fn search(
    cost: &[Vec<f32>],
    current: &mut Vec<usize>,
    used: &mut Vec<bool>,
    total: f32,
    best: &mut Option<Vec<usize>>,
    best_total: &mut f32,
) {
    let i = current.len();
    if i == cost.len() {
        // every row has a column: is this the best so far?
        if total < *best_total {
            *best_total = total;
            *best = Some(current.clone());
        }
        return;
    }
    for j in 0..used.len() {
        if !used[j] && cost[i][j].is_finite() {
            used[j] = true;
            current.push(j);
            search(cost, current, used, total + cost[i][j], best, best_total);
            current.pop();
            used[j] = false;
        }
    }
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
    let m = cost.len();
    let n = if m > 0 { cost[0].len() } else { 0 };
    let mut big = vec![vec![f32::INFINITY; n + m]; m + n];
    for i in 0..m {
        for j in 0..n {
            big[i][j] = cost[i][j]; // top left: the links
        }
        big[i][n + i] = b; // top right: row i ends its track
    }
    for j in 0..n {
        big[m + j][j] = b; // bottom left: column j starts a track
        for i in 0..m {
            big[m + j][n + i] = 0.0; // bottom right: leftover dummies pair up for free
        }
    }
    big
}

/// C5b. Linking with ends and starts: build the augmented matrix, solve it with `solver`
/// (`global_by_search` gives `Option`, so the program passes `hungarian`), and read the links
/// off the top-left block. Row `i` gets `Some(j)` if it links to column `j`, `None` if its
/// track ends.
pub fn link_with_ends(cost: &[Vec<f32>], b: f32, solver: fn(&[Vec<f32>]) -> Vec<usize>) -> Vec<Option<usize>> {
    let m = cost.len();
    let n = if m > 0 { cost[0].len() } else { 0 };
    let assignment = solver(&augmented(cost, b));
    let mut links = vec![None; m];
    for i in 0..m {
        if assignment[i] < n {
            links[i] = Some(assignment[i]);
        }
    }
    links
}
