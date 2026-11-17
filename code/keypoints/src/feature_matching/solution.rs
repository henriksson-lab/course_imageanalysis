// Model solution for the coding exercises of "Matching descriptors".

/// C1. The table of distances: entry `[i][j]` is the Euclidean (L2) distance between
/// descriptor `i` of image 1 and descriptor `j` of image 2.
pub fn distance_table(d1: &[Vec<f32>], d2: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut table = vec![vec![0.0; d2.len()]; d1.len()];
    for i in 0..d1.len() {
        for j in 0..d2.len() {
            let mut sum = 0.0;
            for k in 0..d1[i].len() {
                sum += (d1[i][k] - d2[j][k]) * (d1[i][k] - d2[j][k]);
            }
            table[i][j] = sum.sqrt();
        }
    }
    table
}

/// C2. The nearest and the second-nearest entry of one row of the table: returns
/// `(j, d1, d2)`, the column of the smallest distance, the smallest and the second smallest
/// distance. The row has at least two entries.
pub fn nearest_two(row: &[f32]) -> (usize, f32, f32) {
    let (mut best, mut d1, mut d2) = (0, f32::INFINITY, f32::INFINITY);
    for (j, &d) in row.iter().enumerate() {
        if d < d1 {
            d2 = d1;
            d1 = d;
            best = j;
        } else if d < d2 {
            d2 = d;
        }
    }
    (best, d1, d2)
}

/// C3. The ratio test: for every row `i`, its nearest column `j` is a match if
/// `d1 / d2 < t`. Returns the accepted pairs `(i, j)` in row order.
/// (`t = f32::INFINITY` accepts every nearest neighbour.)
pub fn ratio_test(table: &[Vec<f32>], t: f32) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    for (i, row) in table.iter().enumerate() {
        let (j, d1, d2) = nearest_two(row);
        if d1 / d2 < t {
            matches.push((i, j));
        }
    }
    matches
}

/// C4. The mutual check: keep a match `(i, j)` only if row `i` is also the nearest row of
/// column `j` (the smallest entry of column `j`).
pub fn mutual_check(table: &[Vec<f32>], matches: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut kept = Vec::new();
    for &(i, j) in matches {
        let mut nearest_row = 0;
        for r in 0..table.len() {
            if table[r][j] < table[nearest_row][j] {
                nearest_row = r;
            }
        }
        if nearest_row == i {
            kept.push((i, j));
        }
    }
    kept
}

/// C5. Precision and recall of the matches. `truth[i]` is the true partner of point `i` of
/// image 1 (or `None` if it has none). Precision = correct / accepted; recall = correct /
/// possible, where possible = the number of points with a partner. Returns
/// `(correct, precision, recall)`; precision is 0 if nothing was accepted.
pub fn precision_recall(matches: &[(usize, usize)], truth: &[Option<usize>]) -> (usize, f32, f32) {
    let correct = matches.iter().filter(|&&(i, j)| truth[i] == Some(j)).count();
    let possible = truth.iter().filter(|t| t.is_some()).count();
    let precision = if matches.is_empty() { 0.0 } else { correct as f32 / matches.len() as f32 };
    (correct, precision, correct as f32 / possible as f32)
}
