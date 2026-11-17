// Model solution for the coding exercises of "Hit-or-miss and thinning".

use imgcourse::{Border, Image};

/// A hit-or-miss template, row by row as in the lecture: `t[row][column]`, centre `t[1][1]`.
/// `1` = must be object, `0` = must be background, `DONT_CARE` (−1) = either.
pub type Template = [[i8; 3]; 3];
pub const DONT_CARE: i8 = -1;
const X: i8 = DONT_CARE;

/// End point with its single neighbour above: rows `010 / 010 / 000`.
pub const END_POINT: Template = [[0, 1, 0], [0, 1, 0], [0, 0, 0]];
/// Isolated pixel: rows `000 / 010 / 000`.
pub const ISOLATED: Template = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
/// Golay's L1, an edge pixel: rows `000 / ·1· / 111`.
pub const L1: Template = [[0, 0, 0], [X, 1, X], [1, 1, 1]];
/// Golay's L2, a corner pixel: rows `·00 / 110 / ·1·`.
pub const L2: Template = [[X, 0, 0], [1, 1, 0], [X, 1, X]];

/// The 8 cells around the centre, clockwise from the top-left, as `(row, column)`.
pub const RING: [(usize, usize); 8] = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)];

/// C1. The hit-or-miss transform: true where the template, centred on the pixel, fits
/// completely (every 1-cell on object, every 0-cell on background). Outside the image is
/// background.
pub fn hit_or_miss(img: &Image<bool>, t: &Template) -> Image<bool> {
    img.map_xy(|x, y, _| {
        for row in 0..3 {
            for col in 0..3 {
                let want = t[row][col];
                if want == DONT_CARE {
                    continue;
                }
                let v = img.get_border(x as isize + col as isize - 1, y as isize + row as isize - 1, Border::Zero);
                if v != (want == 1) {
                    return false;
                }
            }
        }
        true
    })
}

/// C2. Turn a template by 45° clockwise: every outer cell moves one step clockwise around
/// the centre (along `RING`); the centre stays.
pub fn rotate45(t: &Template) -> Template {
    let mut out = *t;
    for i in 0..8 {
        let (r0, c0) = RING[i];
        let (r1, c1) = RING[(i + 1) % 8];
        out[r1][c1] = t[r0][c0];
    }
    out
}

/// Turn a template by `eighths` × 45° clockwise (2 = 90°).
pub fn turn(t: &Template, eighths: usize) -> Template {
    let mut out = *t;
    for _ in 0..eighths {
        out = rotate45(&out);
    }
    out
}

/// The union of the hit-or-miss transforms of all 8 turns of `t` (e.g. all end points).
pub fn hit_or_miss_all_turns(img: &Image<bool>, t: &Template) -> Image<bool> {
    let mut out = Image::new(img.width(), img.height(), false);
    for k in 0..8 {
        out = out.zip_map(&hit_or_miss(img, &turn(t, k)), |a, b| a || b);
    }
    out
}

/// C3. Is the object pixel `(x, y)` simple? Among its 8 neighbours there must be exactly one
/// object part (8-connected) and exactly one background part (4-connected) that contains
/// one of the pixel's 4 neighbours. Outside the image is background.
pub fn is_simple(img: &Image<bool>, x: usize, y: usize) -> bool {
    // the 3 × 3 neighbourhood, centre left out
    let mut n = [[false; 3]; 3];
    for (r, row) in n.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = img.get_border(x as isize + c as isize - 1, y as isize + r as isize - 1, Border::Zero);
        }
    }
    // Count the parts of value `value` with flood fills inside the 3 × 3 window, skipping
    // the centre. With `only_touching`, count only parts that contain a 4-neighbour.
    let parts = |value: bool, eight: bool, only_touching: bool| {
        let mut seen = [[false; 3]; 3];
        seen[1][1] = true;
        let mut count = 0;
        for &(r0, c0) in &RING {
            if seen[r0][c0] || n[r0][c0] != value {
                continue;
            }
            let (mut stack, mut touches) = (vec![(r0, c0)], false);
            seen[r0][c0] = true;
            while let Some((r, c)) = stack.pop() {
                touches |= r == 1 || c == 1; // N, E, S or W of the centre
                for &(r2, c2) in &RING {
                    let (dr, dc) = (r2.abs_diff(r), c2.abs_diff(c));
                    let adjacent = if eight { dr <= 1 && dc <= 1 } else { dr + dc == 1 };
                    if adjacent && !seen[r2][c2] && n[r2][c2] == value {
                        seen[r2][c2] = true;
                        stack.push((r2, c2));
                    }
                }
            }
            if touches || !only_touching {
                count += 1;
            }
        }
        count
    };
    parts(true, true, false) == 1 && parts(false, false, true) == 1
}

/// C4. Thinning with Golay's L1 and L2 in 4 turns each, in the order L1, L2, L1 ↻90°,
/// L2 ↻90°, ..., L2 ↻270°. Sequential (`parallel` false): each template removes its matches
/// before the next looks. Parallel: all 8 are tested on the same image and their matches
/// removed together. Repeat passes until nothing changes; returns the skeleton and the
/// number of passes that removed something.
pub fn thin(img: &Image<bool>, parallel: bool) -> (Image<bool>, usize) {
    let mut templates = Vec::new();
    for quarter in 0..4 {
        templates.push(turn(&L1, 2 * quarter));
        templates.push(turn(&L2, 2 * quarter));
    }
    let mut out = img.clone();
    let mut passes = 0;
    loop {
        let before = out.clone();
        if parallel {
            let mut remove = Image::new(out.width(), out.height(), false);
            for t in &templates {
                remove = remove.zip_map(&hit_or_miss(&out, t), |a, b| a || b);
            }
            out = out.zip_map(&remove, |v, r| v && !r);
        } else {
            for t in &templates {
                let hits = hit_or_miss(&out, t);
                out = out.zip_map(&hits, |v, r| v && !r);
            }
        }
        if out == before {
            return (out, passes);
        }
        passes += 1;
    }
}

/// C5. Thinning by simple points: repeat passes until nothing changes. A pass visits the
/// four sides in turn: the border pixels with background above, then below, then to the
/// right, then to the left. For each side, first list its border pixels; then go through
/// the list and remove every pixel that is simple right now, except end points (exactly
/// one object neighbour) when `keep_end_points` is set.
pub fn thin_simple(img: &Image<bool>, keep_end_points: bool) -> Image<bool> {
    let mut out = img.clone();
    let sides = [(0, -1), (0, 1), (1, 0), (-1, 0)];
    loop {
        let mut changed = false;
        for (dx, dy) in sides {
            // the border pixels on this side, chosen before any of them is removed
            let border: Vec<(usize, usize)> =
                out.coords().filter(|&(x, y)| out[(x, y)] && !out.get_border(x as isize + dx, y as isize + dy, Border::Zero)).collect();
            for (x, y) in border {
                let mut neighbours = 0;
                for (ex, ey) in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] {
                    neighbours += out.get_border(x as isize + ex, y as isize + ey, Border::Zero) as usize;
                }
                if keep_end_points && neighbours == 1 {
                    continue;
                }
                if is_simple(&out, x, y) {
                    out[(x, y)] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            return out;
        }
    }
}
