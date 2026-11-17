// Coding exercises for "Spanning trees: single linkage and Felzenszwalb–Huttenlocher".
// Replace each `todo!()` with your code, then check it with `cargo test -p graphseg mst_seg::`
// and look at `cargo run -p graphseg -- mst-seg`.
#![allow(unused_variables, unused_imports)]
// Felzenszwalb–Huttenlocher".

use imgcourse::Image;
use crate::common::sorted;

/// An edge `(a, b, d)` between nodes `a` and `b` with dissimilarity `d`.
pub use crate::common::Edge;

/// The edges of the pixel graph of `img`, with `d = |I_p − I_q|`: to the right and down, and
/// with `eight` also the two diagonals down. Pixel `(x, y)` is node `y · width + x`.
pub fn grid_edges(img: &Image<f32>, eight: bool) -> Vec<Edge> {
    let (w, h) = (img.width(), img.height());
    let mut edges = Vec::new();
    for (x, y) in img.coords() {
        let p = y * w + x;
        let mut add = |qx: usize, qy: usize| edges.push((p, qy * w + qx, (img[(x, y)] - img[(qx, qy)]).abs()));
        if x + 1 < w {
            add(x + 1, y);
        }
        if y + 1 < h {
            add(x, y + 1);
            if eight && x + 1 < w {
                add(x + 1, y + 1);
            }
            if eight && x > 0 {
                add(x - 1, y + 1);
            }
        }
    }
    edges
}

/// C1. Union–find: every node points to a parent; a root (its own parent) represents its
/// component. `size[r]` is the number of nodes in the component of root `r`.
#[derive(Clone, Debug)]
pub struct UnionFind {
    pub parent: Vec<usize>,
    pub size: Vec<usize>,
}

impl UnionFind {
    /// C1. `n` nodes, each its own component.
    pub fn new(n: usize) -> Self {
        todo!("C1: new")
    }

    /// C1. The root of `a`'s component. On the way up, let every node point to its
    /// grandparent ("path halving"), so that later look-ups are short.
    pub fn find(&mut self, a: usize) -> usize {
        todo!("C1: find")
    }

    /// C1. Join the components of `a` and `b` (the smaller one below the larger). Returns the
    /// new root, or `None` if they were already one component.
    pub fn union(&mut self, a: usize, b: usize) -> Option<usize> {
        todo!("C1: union")
    }
}

/// The components as labels `0, 1, 2, …`, numbered in the order of their first node.
pub fn component_labels(uf: &mut UnionFind) -> Vec<usize> {
    let n = uf.parent.len();
    let mut label_of_root = vec![usize::MAX; n];
    let mut next = 0;
    let mut labels = vec![0; n];
    for i in 0..n {
        let r = uf.find(i);
        if label_of_root[r] == usize::MAX {
            label_of_root[r] = next;
            next += 1;
        }
        labels[i] = label_of_root[r];
    }
    labels
}

/// C2. Kruskal's algorithm: the edges of a minimum spanning tree (forest), in the order in
/// which they are joined.
///
/// Hint: `sorted(edges)` (given) sorts by weight, keeping the order of equal weights.
pub fn kruskal(n: usize, edges: &[Edge]) -> Vec<Edge> {
    todo!("C2: kruskal")
}

/// C3. Single linkage at threshold `t`: join across every edge with `d < t`; the regions are
/// the components (labels as in `component_labels`).
///
/// Hint: the order of the edges does not matter here. `component_labels(&mut uf)` (given)
/// turns the union–find into labels.
pub fn single_linkage(n: usize, edges: &[Edge], t: f32) -> Vec<usize> {
    todo!("C3: single_linkage")
}

/// C4. Felzenszwalb–Huttenlocher: Kruskal's order, but join the components `C1` and `C2` of an
/// edge of weight `d` only if `d ≤ min(Int(C1) + k/|C1|, Int(C2) + k/|C2|)`. `Int(C)` is the
/// heaviest edge joined inside `C` (0 for a single node); after a join it is `d`.
///
/// Hint: keep `Int` in a `Vec<f32>` indexed by the root; after a union, store `d` at the new
/// root.
pub fn felzenszwalb(n: usize, edges: &[Edge], k: f32) -> Vec<usize> {
    todo!("C4: felzenszwalb")
}

/// C5. Clean-up to a minimum size: start from the regions `labels`; go through the edges in
/// increasing order and join the two regions of an edge if either has fewer than `min_size`
/// nodes.
///
/// Hint: first join the nodes of each given region (e.g. every node with the first node of
/// its label), then go through `sorted(edges)`.
pub fn merge_small(n: usize, edges: &[Edge], labels: &[usize], min_size: usize) -> Vec<usize> {
    todo!("C5: merge_small")
}
