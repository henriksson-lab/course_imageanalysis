// Model solution for the coding exercises of "Watershed as a minimum spanning forest".

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
    let (w, h) = (f.width(), f.height());
    let mut edges = Vec::new();
    for (x, y) in f.coords() {
        let p = y * w + x;
        if x + 1 < w {
            edges.push((p, p + 1, f[(x, y)].max(f[(x + 1, y)])));
        }
        if y + 1 < h {
            edges.push((p, p + w, f[(x, y)].max(f[(x, y + 1)])));
        }
    }
    edges
}

/// C2. Prim's algorithm from all markers at once (Meyer's flooding on edges). `markers[p]` is
/// 0 for a free node, else the marker's label. Returns the label of every node, and the
/// indices of the cut edges (popped with two different labels at their ends), in pop order.
pub fn prim_from_markers(n: usize, edges: &[Edge], markers: &[u32]) -> (Vec<u32>, Vec<usize>) {
    let inc = incident_edges(n, edges);
    let mut labels = markers.to_vec();
    let mut queue = EdgeQueue::new();
    let other_end = |e: usize, p: usize| if edges[e].0 == p { edges[e].1 } else { edges[e].0 };
    // queue the edges that leave the markers, marker nodes in index order
    for p in 0..n {
        if labels[p] != 0 {
            for &e in &inc[p] {
                if labels[other_end(e, p)] == 0 {
                    queue.push(edges[e].2, e);
                }
            }
        }
    }
    let mut cuts = Vec::new();
    while let Some(e) = queue.pop() {
        let (a, b, _) = edges[e];
        let (la, lb) = (labels[a], labels[b]);
        if la != 0 && lb != 0 {
            if la != lb {
                cuts.push(e); // two trees meet: a watershed edge
            }
            continue;
        }
        // grow: the free end gets the label of the labelled end
        let (new, label) = if la == 0 { (a, lb) } else { (b, la) };
        labels[new] = label;
        for &f in &inc[new] {
            if labels[other_end(f, new)] == 0 {
                queue.push(edges[f].2, f);
            }
        }
    }
    (labels, cuts)
}

/// C3. The same forest by Kruskal with a virtual root: every component carries the label of
/// the marker it contains (0 if none). Go through the edges sorted by weight (equal weights in
/// list order); join two components unless both already carry a marker (then it is a cut
/// edge, as if the edge closed a cycle through the root). Returns the label of every node.
pub fn kruskal_from_markers(n: usize, edges: &[Edge], markers: &[u32]) -> Vec<u32> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut label = markers.to_vec(); // label of the component, stored at its root
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&i, &j| edges[i].2.total_cmp(&edges[j].2));
    for e in order {
        let (ra, rb) = (find(&mut parent, edges[e].0), find(&mut parent, edges[e].1));
        if ra == rb || (label[ra] != 0 && label[rb] != 0) {
            continue;
        }
        parent[rb] = ra;
        label[ra] = label[ra].max(label[rb]); // at most one of them is non-zero
    }
    (0..n).map(|p| label[find(&mut parent, p)]).collect()
}

/// C4. The minimax distance from `source` to every node: the lowest possible highest edge on a
/// path (0 at the source itself; all weights here are at least 0). Dijkstra's algorithm with
/// "the larger of the two" instead of "the sum".
pub fn minimax_distance(n: usize, edges: &[Edge], source: usize) -> Vec<f32> {
    let inc = incident_edges(n, edges);
    let mut dist = vec![f32::INFINITY; n];
    let mut done = vec![false; n];
    dist[source] = 0.0;
    let mut queue = EdgeQueue::new(); // here: (distance, node) instead of (weight, edge)
    queue.push(dist[source], source);
    while let Some(p) = queue.pop() {
        if done[p] {
            continue;
        }
        done[p] = true;
        for &e in &inc[p] {
            let (a, b, d) = edges[e];
            let q = if a == p { b } else { a };
            let via = dist[p].max(d);
            if via < dist[q] {
                dist[q] = via;
                queue.push(via, q);
            }
        }
    }
    dist
}
