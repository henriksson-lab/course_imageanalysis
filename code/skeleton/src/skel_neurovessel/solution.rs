// Model solution for the coding exercises of "Measuring neurons and vessel networks".

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
pub fn tree_counts(tree: &[Point]) -> (f32, usize, usize) {
    let c = children(tree);
    let mut length = 0.0;
    let (mut branch_points, mut tips) = (0, 0);
    for (i, p) in tree.iter().enumerate() {
        if let Some(q) = p.parent {
            length += dist(p, &tree[q]);
            if c[i].len() >= 2 {
                branch_points += 1;
            }
        }
        if c[i].is_empty() {
            tips += 1;
        }
    }
    (length, branch_points, tips)
}

/// C2. The branches: start a chain at the root and at every branch point, once for each
/// child, and follow single children until a point with zero or several children.
pub fn branches(tree: &[Point]) -> Vec<Branch> {
    let c = children(tree);
    let mut out = Vec::new();
    for start in 0..tree.len() {
        let is_node = tree[start].parent.is_none() || c[start].len() >= 2;
        if !is_node {
            continue;
        }
        for &first in &c[start] {
            let mut points = vec![start, first];
            let mut length = dist(&tree[start], &tree[first]);
            let mut current = first;
            while c[current].len() == 1 {
                let next = c[current][0];
                length += dist(&tree[current], &tree[next]);
                points.push(next);
                current = next;
            }
            let chord = dist(&tree[start], &tree[current]);
            out.push(Branch { points, length, chord });
        }
    }
    out
}

/// C3. The Strahler order of every point, i.e. of the segment from it to its parent: a tip
/// has order 1; otherwise take the largest order of its children, plus 1 if two or more
/// children have that largest order. (The root gets the order of the whole tree.)
pub fn strahler(tree: &[Point]) -> Vec<u32> {
    let c = children(tree);
    let mut order = vec![0; tree.len()];
    // children come after their parents, so go backwards
    for i in (0..tree.len()).rev() {
        if c[i].is_empty() {
            order[i] = 1;
            continue;
        }
        let max = c[i].iter().map(|&k| order[k]).max().unwrap();
        let reach = c[i].iter().filter(|&&k| order[k] == max).count();
        order[i] = if reach >= 2 { max + 1 } else { max };
    }
    order
}

/// C4. The Sholl profile: for each radius `r`, the number of segments (point–parent) that
/// cross the circle of radius `r` around `(cx, cy)`: one end closer than `r`, the other at
/// `r` or farther.
pub fn sholl(tree: &[Point], cx: f32, cy: f32, radii: &[f32]) -> Vec<usize> {
    let d = |p: &Point| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt();
    radii
        .iter()
        .map(|&r| {
            tree.iter()
                .filter_map(|p| p.parent.map(|q| (d(p), d(&tree[q]))))
                .filter(|&(a, b)| (a < r) != (b < r))
                .count()
        })
        .collect()
}

/// C5. Match measured points to true points within `tolerance`: take all pairs closer than
/// that, closest first, and accept a pair if neither point is used yet. Returns the matched
/// pairs `(true index, measured index)`, the missed true points and the spurious measured
/// points.
pub fn match_points(truth: &[[f32; 2]], measured: &[[f32; 2]], tolerance: f32) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let mut pairs = Vec::new();
    for (i, t) in truth.iter().enumerate() {
        for (j, m) in measured.iter().enumerate() {
            let d = ((t[0] - m[0]).powi(2) + (t[1] - m[1]).powi(2)).sqrt();
            if d <= tolerance {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut used_t, mut used_m) = (vec![false; truth.len()], vec![false; measured.len()]);
    let mut matched = Vec::new();
    for (_, i, j) in pairs {
        if !used_t[i] && !used_m[j] {
            used_t[i] = true;
            used_m[j] = true;
            matched.push((i, j));
        }
    }
    let missed = (0..truth.len()).filter(|&i| !used_t[i]).collect();
    let spurious = (0..measured.len()).filter(|&j| !used_m[j]).collect();
    (matched, missed, spurious)
}
