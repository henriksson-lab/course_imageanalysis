// Model solution for the coding exercises of "Graphs, cuts and maximum flow".
//
// A graph with n nodes is its capacity matrix: `cap[u][v]` is the capacity of the edge
// u → v (0 if there is none). Flows are matrices of the same size.

use std::collections::VecDeque;

/// An augmenting path: its nodes from s to t, and its bottleneck.
pub type Augmentation = (Vec<usize>, f32);

/// C1. The cost of an s–t cut: the total capacity of the edges from the source side `S` to
/// the sink side `T`. `source_side[u]` is true for the nodes in S. Edges from T back to S
/// do not count.
pub fn cut_cost(cap: &[Vec<f32>], source_side: &[bool]) -> f32 {
    let n = cap.len();
    let mut sum = 0.0;
    for u in 0..n {
        for v in 0..n {
            if source_side[u] && !source_side[v] {
                sum += cap[u][v];
            }
        }
    }
    sum
}

/// C2a. The residual graph of a flow (slide "Augmenting paths and the residual graph"):
/// `r[u][v]` is how much more can go from u to v: the room left on the edge u → v,
/// c(u, v) − f(u, v), plus the flow on v → u that can be undone, f(v, u).
pub fn residual(cap: &[Vec<f32>], flow: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = cap.len();
    let mut r = vec![vec![0.0; n]; n];
    for u in 0..n {
        for v in 0..n {
            r[u][v] = cap[u][v] - flow[u][v] + flow[v][u];
        }
    }
    r
}

/// C2b. A shortest path (fewest edges) from `s` to `t` along residual arcs with room
/// (r > 0), by breadth-first search: the list of nodes from s to t, or `None`.
pub fn shortest_path(r: &[Vec<f32>], s: usize, t: usize) -> Option<Vec<usize>> {
    let n = r.len();
    let mut came_from = vec![None; n];
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    seen[s] = true;
    queue.push_back(s);
    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            if !seen[v] && r[u][v] > 0.0 {
                seen[v] = true;
                came_from[v] = Some(u);
                queue.push_back(v);
            }
        }
    }
    if !seen[t] {
        return None;
    }
    // walk back from t
    let mut path = vec![t];
    let mut v = t;
    while let Some(u) = came_from[v] {
        path.push(u);
        v = u;
    }
    path.reverse();
    Some(path)
}

/// C3. Maximum flow by shortest augmenting paths (Edmonds–Karp): start with flow 0; while
/// the residual graph has a path from s to t, push its bottleneck along it. Returns the
/// final flow and the augmenting paths with their bottlenecks, in order. The value of the
/// flow is the sum of the bottlenecks.
pub fn max_flow(cap: &[Vec<f32>], s: usize, t: usize) -> (Vec<Vec<f32>>, Vec<Augmentation>) {
    let n = cap.len();
    let mut flow = vec![vec![0.0; n]; n];
    let mut paths = Vec::new();
    loop {
        let r = residual(cap, &flow);
        let Some(path) = shortest_path(&r, s, t) else { break };
        // the bottleneck: the smallest residual capacity along the path
        let mut b = f32::INFINITY;
        for k in 1..path.len() {
            b = b.min(r[path[k - 1]][path[k]]);
        }
        for k in 1..path.len() {
            let (u, v) = (path[k - 1], path[k]);
            // first undo flow on v → u (a backward arc), then add the rest on u → v
            let undo = b.min(flow[v][u]);
            flow[v][u] -= undo;
            flow[u][v] += b - undo;
        }
        paths.push((path, b));
    }
    (flow, paths)
}

/// C4. The source side of the minimum cut: the nodes that can still be reached from `s` in
/// the residual graph of the maximum flow (slide "Max-flow = min-cut").
pub fn source_side(cap: &[Vec<f32>], flow: &[Vec<f32>], s: usize) -> Vec<bool> {
    let r = residual(cap, flow);
    let n = cap.len();
    let mut seen = vec![false; n];
    let mut stack = vec![s];
    seen[s] = true;
    while let Some(u) = stack.pop() {
        for v in 0..n {
            if !seen[v] && r[u][v] > 0.0 {
                seen[v] = true;
                stack.push(v);
            }
        }
    }
    seen
}
