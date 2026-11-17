// Coding exercises for "Graphs, cuts and maximum flow". Replace each `todo!()`
// with your code, then check it with `cargo test -p graphcuts maxflow::` and look at `cargo run -p graphcuts -- maxflow`.
#![allow(unused_variables, unused_imports)]

// A graph with n nodes is its capacity matrix: `cap[u][v]` is the capacity of the edge
// u → v (0 if there is none). Flows are matrices of the same size.

use std::collections::VecDeque;

/// An augmenting path: its nodes from s to t, and its bottleneck.
pub type Augmentation = (Vec<usize>, f32);

/// C1. The cost of an s–t cut: the total capacity of the edges from the source side `S` to
/// the sink side `T`. `source_side[u]` is true for the nodes in S. Edges from T back to S
/// do not count.
pub fn cut_cost(cap: &[Vec<f32>], source_side: &[bool]) -> f32 {
    todo!("C1: cut_cost")
}

/// C2a. The residual graph of a flow (slide "Augmenting paths and the residual graph"):
/// `r[u][v]` is how much more can go from u to v: the room left on the edge u → v,
/// c(u, v) − f(u, v), plus the flow on v → u that can be undone, f(v, u).
pub fn residual(cap: &[Vec<f32>], flow: &[Vec<f32>]) -> Vec<Vec<f32>> {
    todo!("C2a: residual")
}

/// C2b. A shortest path (fewest edges) from `s` to `t` along residual arcs with room
/// (r > 0), by breadth-first search: the list of nodes from s to t, or `None`.
///
/// Hint: a queue (`VecDeque`: `push_back`, `pop_front`) of nodes to visit, starting with s;
/// remember for every node the node it was reached from, then walk back from t.
pub fn shortest_path(r: &[Vec<f32>], s: usize, t: usize) -> Option<Vec<usize>> {
    todo!("C2b: shortest_path")
}

/// C3. Maximum flow by shortest augmenting paths (Edmonds–Karp): start with flow 0; while
/// the residual graph has a path from s to t, push its bottleneck along it. Returns the
/// final flow and the augmenting paths with their bottlenecks, in order. The value of the
/// flow is the sum of the bottlenecks.
///
/// Hint: in a loop: `residual`, then `shortest_path` (stop when it is `None`), then the
/// bottleneck = the smallest `r[u][v]` along the path. To push b along u → v, first take as
/// much as possible off `flow[v][u]` (undoing flow), and add the rest to `flow[u][v]`.
pub fn max_flow(cap: &[Vec<f32>], s: usize, t: usize) -> (Vec<Vec<f32>>, Vec<Augmentation>) {
    todo!("C3: max_flow")
}

/// C4. The source side of the minimum cut: the nodes that can still be reached from `s` in
/// the residual graph of the maximum flow (slide "Max-flow = min-cut").
///
/// Hint: the same search as in `shortest_path`, but you only need the `seen` list.
pub fn source_side(cap: &[Vec<f32>], flow: &[Vec<f32>], s: usize) -> Vec<bool> {
    todo!("C4: source_side")
}
