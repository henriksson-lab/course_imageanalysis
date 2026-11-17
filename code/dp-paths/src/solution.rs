// Model solution for the coding exercises of "The DP principle: minimal paths on a grid".

use imgcourse::Image;

/// C1. The DP table of the cheapest monotone path through a cost image: one pixel per row,
/// each step one row down and at most one column sideways (↙ ↓ ↘). Returns the table `M`,
/// where `M(x, y)` is the cost of the cheapest path from the top row to pixel `(x, y)`, and
/// the pointers: the step `d` (−1, 0 or 1) such that the minimum came from `(x + d, y − 1)`.
/// On a tie, straight wins, then left, then right.
pub fn dp_table(cost: &Image<f32>) -> (Image<f32>, Image<isize>) {
    let (w, h) = (cost.width(), cost.height());
    let mut m = Image::new(w, h, 0.0);
    let mut pointer = Image::new(w, h, 0isize);
    // Row 0: the path starts here, M = C.
    for x in 0..w {
        m[(x, 0)] = cost[(x, 0)];
    }
    // Row by row: M(x, y) = C(x, y) + the smallest M of the (up to) three pixels above.
    for y in 1..h {
        for x in 0..w {
            let mut best = f32::INFINITY;
            let mut best_d = 0;
            for d in [0, -1, 1] {
                let xa = x as isize + d;
                if xa < 0 || xa >= w as isize {
                    continue; // outside the image: M = ∞
                }
                let candidate = m[(xa as usize, y - 1)];
                if candidate < best {
                    best = candidate;
                    best_d = d;
                }
            }
            m[(x, y)] = cost[(x, y)] + best;
            pointer[(x, y)] = best_d;
        }
    }
    (m, pointer)
}

/// C2. Backtracking: start at the smallest `M` of the last row (the leftmost one on a tie)
/// and follow the pointers up. Returns the column of the path in every row, from the top
/// row to the bottom row.
pub fn backtrack(m: &Image<f32>, pointer: &Image<isize>) -> Vec<usize> {
    let h = m.height();
    let mut x = 0;
    for xx in 0..m.width() {
        if m[(xx, h - 1)] < m[(x, h - 1)] {
            x = xx;
        }
    }
    let mut columns = vec![0; h];
    for y in (0..h).rev() {
        columns[y] = x;
        if y > 0 {
            x = (x as isize + pointer[(x, y)]) as usize;
        }
    }
    columns
}

/// C3. The greedy path: start at the cheapest pixel of the top row (the leftmost one on a
/// tie), then always step to the cheapest of the (up to) three pixels below (on a tie:
/// straight, then left, then right). Returns the column in every row.
pub fn greedy_path(cost: &Image<f32>) -> Vec<usize> {
    let (w, h) = (cost.width(), cost.height());
    let mut x = 0;
    for xx in 0..w {
        if cost[(xx, 0)] < cost[(x, 0)] {
            x = xx;
        }
    }
    let mut columns = vec![x];
    for y in 1..h {
        let mut best = f32::INFINITY;
        let mut next = x;
        for d in [0, -1, 1] {
            let xb = x as isize + d;
            if xb >= 0 && xb < w as isize && cost[(xb as usize, y)] < best {
                best = cost[(xb as usize, y)];
                next = xb as usize;
            }
        }
        x = next;
        columns.push(x);
    }
    columns
}

/// The cost of a path given by its column in every row: the sum of its pixels.
pub fn path_cost(cost: &Image<f32>, columns: &[usize]) -> f32 {
    columns.iter().enumerate().map(|(y, &x)| cost[(x, y)]).sum()
}

/// C4. The number of monotone paths from the top row to every pixel of a `width × height`
/// image: the same table as C1, with "sum" in place of "cost + min". (In `f64`, because the
/// numbers grow very fast.)
pub fn count_paths(width: usize, height: usize) -> Image<f64> {
    let mut n = Image::new(width, height, 0.0);
    for x in 0..width {
        n[(x, 0)] = 1.0; // one path: the pixel itself
    }
    for y in 1..height {
        for x in 0..width {
            let mut sum = 0.0;
            for d in [-1, 0, 1] {
                let xa = x as isize + d;
                if xa >= 0 && xa < width as isize {
                    sum += n[(xa as usize, y - 1)];
                }
            }
            n[(x, y)] = sum;
        }
    }
    n
}

/// C5a. The forward pass of the city-block distance transform: visit the pixels row by row,
/// left to right, and set `D(x, y) = min(D(x, y), D(x, y − 1) + 1, D(x − 1, y) + 1)`.
/// `D` starts at 0 on the background pixels (`true`) and ∞ elsewhere.
pub fn distance_forward(background: &Image<bool>) -> Image<f32> {
    let mut d = background.map(|b| if b { 0.0 } else { f32::INFINITY });
    for y in 0..d.height() {
        for x in 0..d.width() {
            if y > 0 {
                d[(x, y)] = d[(x, y)].min(d[(x, y - 1)] + 1.0);
            }
            if x > 0 {
                d[(x, y)] = d[(x, y)].min(d[(x - 1, y)] + 1.0);
            }
        }
    }
    d
}

/// C5b. The full city-block distance transform: the forward pass, then a backward pass in
/// the opposite order (bottom to top, right to left) that looks down and right.
pub fn distance_two_passes(background: &Image<bool>) -> Image<f32> {
    let mut d = distance_forward(background);
    let (w, h) = (d.width(), d.height());
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            if y + 1 < h {
                d[(x, y)] = d[(x, y)].min(d[(x, y + 1)] + 1.0);
            }
            if x + 1 < w {
                d[(x, y)] = d[(x, y)].min(d[(x + 1, y)] + 1.0);
            }
        }
    }
    d
}
