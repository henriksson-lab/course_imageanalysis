// Coding exercises for "Watershed as a minimum spanning forest".
// Replace each `todo!()` with your code, then check it with `cargo test -p graphseg ws_msf::`
// and look at `cargo run -p graphseg -- ws-msf`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use crate::common::find;

/// An edge `(a, b, d)` between nodes `a` and `b` with weight (altitude) `d`.
pub use crate::common::Edge;

/// A priority queue of edges (by their index in an edge list): `pop` gives the edge with the
/// smallest weight, and among equal weights the one pushed first (first in, first out).
#[derive(Default)]
pub struct EdgeQueue {
    heap: BinaryHeap<Entry>,
    pushed: u64,
}

struct Entry {
    weight: f32,
    order: u64,
    edge: usize,
}

impl Ord for Entry {
    // reversed, so that the largest entry of the heap is the smallest weight, earliest push
    fn cmp(&self, other: &Self) -> Ordering {
        other.weight.total_cmp(&self.weight).then(other.order.cmp(&self.order))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}

impl EdgeQueue {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, weight: f32, edge: usize) {
        self.heap.push(Entry { weight, order: self.pushed, edge });
        self.pushed += 1;
    }
    /// The index of the cheapest edge (first pushed among equals), or `None` when empty.
    pub fn pop(&mut self) -> Option<usize> {
        self.heap.pop().map(|e| e.edge)
    }
}

/// For every node, the indices of its edges, in the order of the edge list.
pub fn incident_edges(n: usize, edges: &[Edge]) -> Vec<Vec<usize>> {
    let mut inc = vec![Vec::new(); n];
    for (i, &(a, b, _)) in edges.iter().enumerate() {
        inc[a].push(i);
        inc[b].push(i);
    }
    inc
}

/// C1. The edges of the pixel graph of a landscape `f` (4-neighbours: for every pixel row by
/// row, first the edge to the right, then the one down), each with the altitude of the pass
/// between its pixels, `d = max(f_p, f_q)`. Pixel `(x, y)` is node `y · width + x`.
pub fn altitude_edges(f: &Image<f32>) -> Vec<Edge> {
    todo!("C1: altitude_edges")
}

/// C2. Prim's algorithm from all markers at once (Meyer's flooding on edges). `markers[p]` is
/// 0 for a free node, else the marker's label. Returns the label of every node, and the
/// indices of the cut edges (popped with two different labels at their ends), in pop order.
///
/// Hint: `incident_edges` (given) lists the edges of every node; `EdgeQueue` (given) pops
/// the cheapest edge, first in first out among equal weights. Queue an edge when one end gets
/// a label and the other end has none yet.
pub fn prim_from_markers(n: usize, edges: &[Edge], markers: &[u32]) -> (Vec<u32>, Vec<usize>) {
    todo!("C2: prim_from_markers")
}

/// C3. The same forest by Kruskal with a virtual root: every component carries the label of
/// the marker it contains (0 if none). Go through the edges sorted by weight (equal weights in
/// list order); join two components unless both already carry a marker (then it is a cut
/// edge, as if the edge closed a cycle through the root). Returns the label of every node.
///
/// Hint: keep a `parent` vector for union–find (`find` is given) and a label per root.
pub fn kruskal_from_markers(n: usize, edges: &[Edge], markers: &[u32]) -> Vec<u32> {
    todo!("C3: kruskal_from_markers")
}

/// C4. The minimax distance from `source` to every node: the lowest possible highest edge on a
/// path (0 at the source itself; all weights here are at least 0). Dijkstra's algorithm with
/// "the larger of the two" instead of "the sum".
///
/// Hint: an `EdgeQueue` also works with (distance, node) instead of (weight, edge). Skip a
/// node when it is popped a second time.
pub fn minimax_distance(n: usize, edges: &[Edge], source: usize) -> Vec<f32> {
    todo!("C4: minimax_distance")
}
