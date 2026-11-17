// Model solution for the coding exercises of "Spanning trees: single linkage and
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
        UnionFind { parent: (0..n).collect(), size: vec![1; n] }
    }

    /// C1. The root of `a`'s component. On the way up, let every node point to its
    /// grandparent ("path halving"), so that later look-ups are short.
    pub fn find(&mut self, a: usize) -> usize {
        let mut a = a;
        while self.parent[a] != a {
            self.parent[a] = self.parent[self.parent[a]];
            a = self.parent[a];
        }
        a
    }

    /// C1. Join the components of `a` and `b` (the smaller one below the larger). Returns the
    /// new root, or `None` if they were already one component.
    pub fn union(&mut self, a: usize, b: usize) -> Option<usize> {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return None;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        Some(ra)
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
pub fn kruskal(n: usize, edges: &[Edge]) -> Vec<Edge> {
    let mut uf = UnionFind::new(n);
    let mut tree = Vec::new();
    for e in sorted(edges) {
        if uf.union(e.0, e.1).is_some() {
            tree.push(e);
        }
    }
    tree
}

/// C3. Single linkage at threshold `t`: join across every edge with `d < t`; the regions are
/// the components (labels as in `component_labels`).
pub fn single_linkage(n: usize, edges: &[Edge], t: f32) -> Vec<usize> {
    let mut uf = UnionFind::new(n);
    for &(a, b, d) in edges {
        if d < t {
            uf.union(a, b);
        }
    }
    component_labels(&mut uf)
}

/// C4. Felzenszwalb–Huttenlocher: Kruskal's order, but join the components `C1` and `C2` of an
/// edge of weight `d` only if `d ≤ min(Int(C1) + k/|C1|, Int(C2) + k/|C2|)`. `Int(C)` is the
/// heaviest edge joined inside `C` (0 for a single node); after a join it is `d`.
pub fn felzenszwalb(n: usize, edges: &[Edge], k: f32) -> Vec<usize> {
    let mut uf = UnionFind::new(n);
    let mut int = vec![0.0f32; n]; // Int of the component, stored at its root
    for (a, b, d) in sorted(edges) {
        let (ra, rb) = (uf.find(a), uf.find(b));
        if ra == rb {
            continue;
        }
        let threshold_a = int[ra] + k / uf.size[ra] as f32;
        let threshold_b = int[rb] + k / uf.size[rb] as f32;
        if d <= threshold_a.min(threshold_b) {
            let r = uf.union(ra, rb).unwrap();
            int[r] = d; // edges come in increasing order: d is the heaviest so far
        }
    }
    component_labels(&mut uf)
}

/// C5. Clean-up to a minimum size: start from the regions `labels`; go through the edges in
/// increasing order and join the two regions of an edge if either has fewer than `min_size`
/// nodes.
pub fn merge_small(n: usize, edges: &[Edge], labels: &[usize], min_size: usize) -> Vec<usize> {
    let mut uf = UnionFind::new(n);
    // first rebuild the given regions
    let mut first_node = vec![usize::MAX; n];
    for i in 0..n {
        if first_node[labels[i]] == usize::MAX {
            first_node[labels[i]] = i;
        } else {
            uf.union(first_node[labels[i]], i);
        }
    }
    for (a, b, _) in sorted(edges) {
        let (ra, rb) = (uf.find(a), uf.find(b));
        if ra != rb && (uf.size[ra] < min_size || uf.size[rb] < min_size) {
            uf.union(ra, rb);
        }
    }
    component_labels(&mut uf)
}
