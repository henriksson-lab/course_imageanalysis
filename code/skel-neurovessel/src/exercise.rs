// Coding exercises for "Measuring neurons and vessel networks". Replace each `todo!()` with
// your code, then check it with `cargo test -p skel-neurovessel` and look at
// `cargo run -p skel-neurovessel`.
#![allow(unused_variables, unused_imports)]

/// One point of a traced tree (slide "Tracing: what neuroscientists use"): a position, a
/// radius and the index of its parent point. The root (the soma) has no parent; every other
/// point comes after its parent in the list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub parent: Option<usize>,
}

/// A branch: the chain of points from a node (the root or a branch point) to the next node
/// (a branch point or a tip), `points[0]` being the first node.
#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    pub points: Vec<usize>,
    /// Path length: the sum of the segment lengths along the chain.
    pub length: f32,
    /// Straight-line distance between the two ends.
    pub chord: f32,
}

impl Branch {
    pub fn tortuosity(&self) -> f32 {
        self.length / self.chord
    }
}

/// The distance between two points.
pub fn dist(a: &Point, b: &Point) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// The children of every point.
pub fn children(tree: &[Point]) -> Vec<Vec<usize>> {
    let mut c = vec![Vec::new(); tree.len()];
    for (i, p) in tree.iter().enumerate() {
        if let Some(q) = p.parent {
            c[q].push(i);
        }
    }
    c
}

/// C1. Total length (the sum of the distances from every point to its parent), the number of
/// branch points (points other than the root with two or more children) and the number of
/// tips (points without children). Returns `(length, branch points, tips)`.
///
/// Hint: `children(tree)` gives the children of every point.
pub fn tree_counts(tree: &[Point]) -> (f32, usize, usize) {
    todo!("C1: tree_counts")
}

/// C2. The branches: start a chain at the root and at every branch point, once for each
/// child, and follow single children until a point with zero or several children.
pub fn branches(tree: &[Point]) -> Vec<Branch> {
    todo!("C2: branches")
}

/// C3. The Strahler order of every point, i.e. of the segment from it to its parent: a tip
/// has order 1; otherwise take the largest order of its children, plus 1 if two or more
/// children have that largest order. (The root gets the order of the whole tree.)
///
/// Hint: children come after their parents in the list, so going through the points from
/// the last to the first, every child's order is known before its parent's.
pub fn strahler(tree: &[Point]) -> Vec<u32> {
    todo!("C3: strahler")
}

/// C4. The Sholl profile: for each radius `r`, the number of segments (point–parent) that
/// cross the circle of radius `r` around `(cx, cy)`: one end closer than `r`, the other at
/// `r` or farther.
///
/// Hint: a segment crosses the circle when `(d1 < r) != (d2 < r)` for the distances `d1`,
/// `d2` of its two ends from the centre.
pub fn sholl(tree: &[Point], cx: f32, cy: f32, radii: &[f32]) -> Vec<usize> {
    todo!("C4: sholl")
}

/// C5. Match measured points to true points within `tolerance`: take all pairs closer than
/// that, closest first, and accept a pair if neither point is used yet. Returns the matched
/// pairs `(true index, measured index)`, the missed true points and the spurious measured
/// points.
///
/// Hint: collect `(distance, i, j)` for all close pairs, sort them with
/// `sort_by(|a, b| a.0.total_cmp(&b.0))`, and keep two lists of booleans for "used".
pub fn match_points(truth: &[[f32; 2]], measured: &[[f32; 2]], tolerance: f32) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    todo!("C5: match_points")
}
