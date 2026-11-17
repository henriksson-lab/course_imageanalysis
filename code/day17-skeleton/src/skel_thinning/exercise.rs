// Coding exercises for "Hit-or-miss and thinning". Replace each `todo!()` with your code,
// then check it with `cargo test -p day17 skel_thinning::` and look at `cargo run -p day17 -- skel-thinning`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `img.get_border(x as isize + col as isize - 1, y as isize + row as isize - 1, Border::Zero)`
/// reads the image under template cell `t[row][col]`; return false at the first cell that fails.
pub fn hit_or_miss(img: &Image<bool>, t: &Template) -> Image<bool> {
    todo!("C1: hit_or_miss")
}

/// C2. Turn a template by 45° clockwise: every outer cell moves one step clockwise around
/// the centre (along `RING`); the centre stays.
///
/// Hint: start from a copy `let mut out = *t;` and set `out` at `RING[i + 1]` (wrapping
/// around after 7) to the value of `t` at `RING[i]`.
pub fn rotate45(t: &Template) -> Template {
    todo!("C2: rotate45")
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
///
/// Hint: copy the 3 × 3 neighbourhood into `[[bool; 3]; 3]`. Count parts with a flood fill
/// (a stack of cells still to visit) over the 8 cells of `RING`: two cells are 8-adjacent if
/// their rows and columns differ by at most 1, 4-adjacent if exactly one differs, by 1. A
/// part touches a 4-neighbour of the centre if one of its cells has row 1 or column 1.
pub fn is_simple(img: &Image<bool>, x: usize, y: usize) -> bool {
    todo!("C3: is_simple")
}

/// C4. Thinning with Golay's L1 and L2 in 4 turns each, in the order L1, L2, L1 ↻90°,
/// L2 ↻90°, ..., L2 ↻270°. Sequential (`parallel` false): each template removes its matches
/// before the next looks. Parallel: all 8 are tested on the same image and their matches
/// removed together. Repeat passes until nothing changes; returns the skeleton and the
/// number of passes that removed something.
///
/// Hint: build the list of 8 templates with `turn(&L1, 2 * quarter)` and `turn(&L2, 2 * quarter)`.
/// Removing the hits: `out.zip_map(&hits, |v, r| v && !r)`. Images can be compared with `==`.
pub fn thin(img: &Image<bool>, parallel: bool) -> (Image<bool>, usize) {
    todo!("C4: thin")
}

/// C5. Thinning by simple points: repeat passes until nothing changes. A pass visits the
/// four sides in turn: the border pixels with background above, then below, then to the
/// right, then to the left. For each side, first list its border pixels; then go through
/// the list and remove every pixel that is simple right now, except end points (exactly
/// one object neighbour) when `keep_end_points` is set.
///
/// Hint: change `out` in place (`out[(x, y)] = false`), so that every test sees the pixels
/// removed before it. Listing the border pixels first matters: otherwise one pass would
/// peel layer after layer from the same side.
pub fn thin_simple(img: &Image<bool>, keep_end_points: bool) -> Image<bool> {
    todo!("C5: thin_simple")
}
