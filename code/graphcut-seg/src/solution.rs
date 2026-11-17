// Model solution for the coding exercises of "Graph-cut segmentation".

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
pub fn contrast_weight(ip: f32, iq: f32, sigma: f32) -> f32 {
    if sigma.is_infinite() {
        return 1.0;
    }
    (-(ip - iq).powi(2) / (2.0 * sigma * sigma)).exp()
}

/// C2. The graph of an image (slide "The graph"): node `y · width + x` for pixel (x, y),
/// then s = width · height and t = s + 1. t-links s → p with capacity D_p(0) and p → t
/// with capacity D_p(1); n-links p → q and q → p, each with capacity λ · w_pq, with w_pq
/// the contrast weight times the neighbourhood weight.
pub fn build_graph(img: &Image<f32>, d0: &Image<f32>, d1: &Image<f32>, lambda: f32, sigma: f32, nb: Neighbours) -> FlowGraph {
    let (w, h) = (img.width(), img.height());
    let s = w * h;
    let t = s + 1;
    let mut g = FlowGraph::new(w * h + 2);
    // each neighbour pair once: (dx, dy, weight)
    let offsets: Vec<(isize, isize, f32)> = match nb {
        Neighbours::Four => vec![(1, 0, 1.0), (0, 1, 1.0)],
        Neighbours::Eight => {
            let diag = 1.0 / (2.0 * 2f32.sqrt());
            vec![(1, 0, 0.5), (0, 1, 0.5), (1, 1, diag), (-1, 1, diag)]
        }
    };
    for (x, y) in img.coords() {
        let p = y * w + x;
        g.add_edge(s, p, d0[(x, y)]);
        g.add_edge(p, t, d1[(x, y)]);
        for &(dx, dy, weight) in &offsets {
            let (qx, qy) = (x as isize + dx, y as isize + dy);
            if img.contains(qx, qy) {
                let (qx, qy) = (qx as usize, qy as usize);
                let c = lambda * weight * contrast_weight(img[(x, y)], img[(qx, qy)], sigma);
                let q = qy * w + qx;
                g.add_edge(p, q, c);
                g.add_edge(q, p, c);
            }
        }
    }
    g
}

/// C3. Segment by a minimum cut: build the graph, compute the maximum flow from s to t, and
/// label the pixels on the source side 1 (object), the others 0. Returns the labels and the
/// value of the flow (= the cost of the cut = the energy of the labelling).
pub fn graph_cut(img: &Image<f32>, d0: &Image<f32>, d1: &Image<f32>, lambda: f32, sigma: f32, nb: Neighbours) -> (Image<u32>, f32) {
    let mut g = build_graph(img, d0, d1, lambda, sigma, nb);
    let s = img.width() * img.height();
    let flow = g.max_flow(s, s + 1);
    let side = g.source_side(s);
    let labels = img.map_xy(|x, y, _| if side[y * img.width() + x] { 1 } else { 0 });
    (labels, flow)
}

/// C4a. The mean and standard deviation of the grey levels of the pixels where
/// `seeds` equals `label` (1: object seeds, 2: background seeds).
pub fn seed_model(img: &Image<f32>, seeds: &Image<u32>, label: u32) -> (f32, f32) {
    let values: Vec<f32> = img.coords().filter(|&(x, y)| seeds[(x, y)] == label).map(|(x, y)| img[(x, y)]).collect();
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// C4b. Data costs from the seeds (slide "Seeds"): fit a Gaussian to the object seeds and one
/// to the background seeds, and take D(ℓ) = (I − μℓ)² / 2sℓ² + log(sℓ / s_min): minus the log
/// of the Gaussian, shifted by the same constant for both labels (log s_min, the smaller of
/// the two) so that no capacity is negative. Hard constraints: an object seed gets D(0) = ∞
/// (it can never be background), a background seed D(1) = ∞.
pub fn seeded_costs(img: &Image<f32>, seeds: &Image<u32>) -> (Image<f32>, Image<f32>) {
    let (mu1, s1) = seed_model(img, seeds, 1);
    let (mu0, s0) = seed_model(img, seeds, 2);
    let s_min = s0.min(s1);
    let cost = |v: f32, mu: f32, s: f32| (v - mu).powi(2) / (2.0 * s * s) + (s / s_min).ln();
    let d0 = img.map_xy(|x, y, v| if seeds[(x, y)] == 1 { f32::INFINITY } else { cost(v, mu0, s0) });
    let d1 = img.map_xy(|x, y, v| if seeds[(x, y)] == 2 { f32::INFINITY } else { cost(v, mu1, s1) });
    (d0, d1)
}
