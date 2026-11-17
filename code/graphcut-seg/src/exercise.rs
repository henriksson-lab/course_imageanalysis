// Coding exercises for "Graph-cut segmentation". Replace each `todo!()`
// with your code, then check it with `cargo test -p graphcut-seg` and look at `cargo run -p graphcut-seg`.
#![allow(unused_variables, unused_imports)]

use crate::FlowGraph;
use imgcourse::Image;

/// Which neighbours are joined by n-links.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Neighbours {
    /// left, right, up, down; weight 1
    Four,
    /// also the diagonals; weights ½ (sides) and 1/(2√2) (diagonals), slide "Failure modes"
    Eight,
}

/// C1. The contrast-sensitive weight of an n-link between grey levels `ip` and `iq`:
/// exp(−(ip − iq)² / 2σ²). For σ = infinity (`f32::INFINITY`) the weight is 1.
///
/// Hint: `sigma.is_infinite()` tells you whether σ = ∞.
pub fn contrast_weight(ip: f32, iq: f32, sigma: f32) -> f32 {
    todo!("C1: contrast_weight")
}

/// C2. The graph of an image (slide "The graph"): node `y · width + x` for pixel (x, y),
/// then s = width · height and t = s + 1. t-links s → p with capacity D_p(0) and p → t
/// with capacity D_p(1); n-links p → q and q → p, each with capacity λ · w_pq, with w_pq
/// the contrast weight times the neighbourhood weight.
///
/// Hint: `FlowGraph::new(n)` and `g.add_edge(u, v, capacity)`. Visit every pixel, add its two
/// t-links, and n-links to the neighbours to the right and below (for 8 neighbours also the
/// two lower diagonals), so that every pair is added once, in both directions.
pub fn build_graph(img: &Image<f32>, d0: &Image<f32>, d1: &Image<f32>, lambda: f32, sigma: f32, nb: Neighbours) -> FlowGraph {
    todo!("C2: build_graph")
}

/// C3. Segment by a minimum cut: build the graph, compute the maximum flow from s to t, and
/// label the pixels on the source side 1 (object), the others 0. Returns the labels and the
/// value of the flow (= the cost of the cut = the energy of the labelling).
///
/// Hint: `g.max_flow(s, t)` returns the flow; afterwards `g.source_side(s)[p]` says whether
/// node p is on the source side.
pub fn graph_cut(img: &Image<f32>, d0: &Image<f32>, d1: &Image<f32>, lambda: f32, sigma: f32, nb: Neighbours) -> (Image<u32>, f32) {
    todo!("C3: graph_cut")
}

/// C4a. The mean and standard deviation of the grey levels of the pixels where
/// `seeds` equals `label` (1: object seeds, 2: background seeds).
pub fn seed_model(img: &Image<f32>, seeds: &Image<u32>, label: u32) -> (f32, f32) {
    todo!("C4a: seed_model")
}

/// C4b. Data costs from the seeds (slide "Seeds"): fit a Gaussian to the object seeds and one
/// to the background seeds, and take D(ℓ) = (I − μℓ)² / 2sℓ² + log(sℓ / s_min): minus the log
/// of the Gaussian, shifted by the same constant for both labels (log s_min, the smaller of
/// the two) so that no capacity is negative. Hard constraints: an object seed gets D(0) = ∞
/// (it can never be background), a background seed D(1) = ∞.
///
/// Hint: compute the two models with `seed_model`, then the two cost images with `img.map_xy`.
pub fn seeded_costs(img: &Image<f32>, seeds: &Image<u32>) -> (Image<f32>, Image<f32>) {
    todo!("C4b: seeded_costs")
}
