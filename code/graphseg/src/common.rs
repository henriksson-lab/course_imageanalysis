//! Helpers shared by the units of day 30 (filled in, not exercises): the edge type, union–find
//! `find`, sorting edges by weight, and the gradient landscape used by the programs.

use imgcourse::{Border, Image};

/// An edge `(a, b, d)` between nodes `a` and `b` with weight `d`.
pub type Edge = (usize, usize, f32);

/// The root of `a` in a union–find forest `parent` (with path halving).
pub fn find(parent: &mut [usize], a: usize) -> usize {
    let mut a = a;
    while parent[a] != a {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    a
}

/// The edges sorted by weight; edges of equal weight keep their order (a stable sort).
pub fn sorted(edges: &[Edge]) -> Vec<Edge> {
    let mut e = edges.to_vec();
    e.sort_by(|a, b| a.2.total_cmp(&b.2));
    e
}

/// The gradient magnitude (Sobel) after a 3 × 3 binomial blur: the landscape to flood.
pub fn gradient(img: &Image<f32>) -> Image<f32> {
    let k = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];
    let blur = img.map_xy(|x, y, _| {
        let mut s = 0.0;
        for dy in 0..3 {
            for dx in 0..3 {
                s += k[dy][dx] / 16.0 * img.get_border(x as isize + dx as isize - 1, y as isize + dy as isize - 1, Border::Clamp);
            }
        }
        s
    });
    blur.map_xy(|x, y, _| {
        let v = |dx: isize, dy: isize| blur.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
        let gx = v(1, -1) + 2.0 * v(1, 0) + v(1, 1) - v(-1, -1) - 2.0 * v(-1, 0) - v(-1, 1);
        let gy = v(-1, 1) + 2.0 * v(0, 1) + v(1, 1) - v(-1, -1) - 2.0 * v(0, -1) - v(1, -1);
        (gx * gx + gy * gy).sqrt()
    })
}
