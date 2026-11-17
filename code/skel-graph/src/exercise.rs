// Coding exercises for "From skeleton to graph: pruning and length". Replace each `todo!()`
// with your code, then check it with `cargo test -p skel-graph` and look at `cargo run -p skel-graph`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// The 8 neighbours of a pixel, as offsets `(dx, dy)`.
pub const N8: [(isize, isize); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// A node of the skeleton graph: an end point, a cluster of touching junction pixels, an
/// isolated pixel, or (for a closed loop without any node) one pixel of the loop.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub pixels: Vec<(usize, usize)>,
}

/// An edge: it joins node `a` to node `b` (`a == b` for a loop) along `path`, the pixels in
/// order from a pixel of node `a` to a pixel of node `b`.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub path: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// The step counts of a path: axial steps `ne`, diagonal steps `no`, corners `nc` (changes
/// of step direction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Steps {
    pub ne: usize,
    pub no: usize,
    pub nc: usize,
}

impl Steps {
    pub fn pixel_steps(&self) -> f32 {
        (self.ne + self.no) as f32
    }
    pub fn weighted(&self) -> f32 {
        self.ne as f32 + std::f32::consts::SQRT_2 * self.no as f32
    }
    /// The corrected estimator of Vossepoel and Smeulders.
    pub fn corrected(&self) -> f32 {
        0.980 * self.ne as f32 + 1.406 * self.no as f32 - 0.091 * self.nc as f32
    }
}

/// C1. The number of skeleton neighbours (8-neighbourhood) of every skeleton pixel; 0 off
/// the skeleton. 1 = end point, 2 = regular, 3 or more = junction.
pub fn neighbour_count(skel: &Image<bool>) -> Image<u8> {
    todo!("C1: neighbour_count")
}

/// C2. Step counts of a path of 8-connected pixels.
///
/// Hint: look at the consecutive pairs with `path.windows(2)`; a step `(dx, dy)` is axial if
/// one of them is 0. Remember the previous step to count the corners.
pub fn path_steps(path: &[(usize, usize)]) -> Steps {
    todo!("C2: path_steps")
}

/// C3. Build the skeleton graph.
///
/// 1. Node pixels are the skeleton pixels with a neighbour count other than 2. Touching
///    junction pixels (count ≥ 3, 8-connected) form one node; every other node pixel is a
///    node on its own.
/// 2. Edges: from every node pixel, step to each neighbouring skeleton pixel. If that is a
///    pixel of another node, the two nodes are joined directly (one edge per pair of nodes).
///    If it is a regular pixel not yet visited, walk along the regular pixels (each has
///    exactly one way on) until a node pixel is reached. (A regular pixel whose two
///    neighbours both belong to the starting node is part of that node: no edge.)
/// 3. Regular pixels still not visited lie on closed loops without nodes: make one of
///    their pixels a node and walk round the loop back to it.
///
/// Hint: keep an image `node_of: Image<Option<usize>>` that says which node a pixel belongs
/// to, and an image `visited` for the regular pixels already on an edge. Write the walk as a
/// loop: from `current`, go to the skeleton neighbour that is not `previous` (use
/// `neighbours_on`), until that neighbour has `node_of` set. The order of nodes and edges
/// does not matter to the checks.
pub fn build_graph(skel: &Image<bool>) -> Graph {
    todo!("C3: build_graph")
}

/// The skeleton pixels among the 8 neighbours of `(x, y)`.
pub fn neighbours_on(skel: &Image<bool>, x: usize, y: usize) -> Vec<(usize, usize)> {
    N8.iter()
        .map(|&(dx, dy)| (x as isize + dx, y as isize + dy))
        .filter(|&(nx, ny)| skel.contains(nx, ny) && skel[(nx as usize, ny as usize)])
        .map(|(nx, ny)| (nx as usize, ny as usize))
        .collect()
}

/// C4. The cyclomatic number E − V + C: edges minus nodes plus connected pieces of the
/// graph (= the number of independent cycles = holes).
///
/// Hint: count the pieces by "union–find": `parent[i]` starts as `i`; for every edge, find
/// the root of both ends (follow `parent` until `parent[i] == i`) and point one root at the
/// other. The pieces are the nodes that are still their own root.
pub fn cycles(g: &Graph) -> isize {
    todo!("C4: cycles")
}

/// C5. Prune spurs: in two rounds, build the graph, find the terminal edges (one end a
/// single-pixel node with one edge, the other end a node with three or more edges) whose
/// weighted length is below `k`, and delete their pixels except the junction's. At each
/// junction, delete the shortest ones first, and stop when only two of its edges are left.
///
/// Hint: the degree of a node is the number of edge ends at it (a loop counts twice). Sort
/// the short spurs of a junction with `sort_by(|a, b| a.0.total_cmp(&b.0))`.
pub fn prune(skel: &Image<bool>, k: f32) -> Image<bool> {
    todo!("C5: prune")
}
