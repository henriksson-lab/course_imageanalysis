// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p graphcuts maxflow::

use imgcourse::Rng;
use graphcuts::maxflow::*;

// Node numbers: s = 0, t = 1, a = 2, b = 3, c = 4, d = 5 (as in the demo).
const S: usize = 0;
const T: usize = 1;

fn graph(n: usize, edges: &[(usize, usize, f32)]) -> Vec<Vec<f32>> {
    let mut cap = vec![vec![0.0; n]; n];
    for &(u, v, c) in edges {
        cap[u][v] = c;
    }
    cap
}

/// The graph of B1–B5.
fn b1() -> Vec<Vec<f32>> {
    graph(6, &[(0, 2, 8.0), (0, 3, 4.0), (2, 3, 8.0), (2, 4, 5.0), (3, 5, 4.0), (4, 5, 7.0), (4, 1, 4.0), (5, 1, 7.0)])
}

fn value(paths: &[(Vec<usize>, f32)]) -> f32 {
    paths.iter().map(|p| p.1).sum()
}

fn side(nodes: &[usize]) -> Vec<bool> {
    (0..6).map(|u| nodes.contains(&u)).collect()
}

#[test]
fn c1_cuts_b1() {
    let cap = b1();
    // {s}, {s, a}, {s, b}, {s, a, b}, {s, a, b, c, d}
    for (nodes, cost) in [(vec![S], 12.0), (vec![S, 2], 17.0), (vec![S, 3], 12.0), (vec![S, 2, 3], 9.0), (vec![S, 2, 3, 4, 5], 11.0)] {
        let c = cut_cost(&cap, &side(&nodes));
        assert!((c - cost).abs() < 1e-5, "B1: cut with source side {nodes:?}: got {c}, expected {cost}");
    }
}

#[test]
fn c2_residual_and_path() {
    let cap = graph(3, &[(0, 2, 5.0), (2, 1, 3.0)]);
    let mut flow = vec![vec![0.0; 3]; 3];
    flow[0][2] = 2.0;
    let r = residual(&cap, &flow);
    assert_eq!((r[0][2], r[2][0]), (3.0, 2.0), "s → a with capacity 5 and flow 2: room 3 forward, 2 backward");
    assert_eq!(shortest_path(&r, 0, 1), Some(vec![0, 2, 1]), "the path s → a → t");
    let full = vec![vec![0.0, 0.0, 5.0], vec![0.0; 3], vec![0.0; 3]];
    let r = residual(&cap, &full);
    assert!(r[0][2] == 0.0 && shortest_path(&r, 0, 1).is_none(), "with s → a full, t cannot be reached");
}

#[test]
fn c3_max_flow_b2() {
    let cap = b1();
    let (flow, paths) = max_flow(&cap, S, T);
    assert!((value(&paths) - 9.0).abs() < 1e-5, "B2: maximum flow {}, expected 9", value(&paths));
    assert_eq!(paths.len(), 3, "B2: three augmenting paths (4, 4, 1)");
    let expected = [(0, 2, 5.0), (0, 3, 4.0), (2, 3, 0.0), (2, 4, 5.0), (3, 5, 4.0), (4, 5, 1.0), (4, 1, 4.0), (5, 1, 5.0)];
    for (u, v, f) in expected {
        assert!((flow[u][v] - f).abs() < 1e-5, "B2: flow on edge {u} → {v}: got {}, expected {f}", flow[u][v]);
    }
}

#[test]
fn c3_max_flow_b5() {
    for (c, expected) in [(6.0, 10.0), (8.0, 11.0)] {
        let mut cap = b1();
        cap[2][4] = c;
        let (_, paths) = max_flow(&cap, S, T);
        assert!((value(&paths) - expected).abs() < 1e-5, "B5: c(a, c) = {c}: flow {}, expected {expected}", value(&paths));
    }
}

#[test]
fn c3_needs_backward_arcs_b6() {
    // B6: s → a, s → b, a → b, a → t, b → t, all capacity 1 (s = 0, t = 1, a = 2, b = 3)
    let cap = graph(4, &[(0, 2, 1.0), (0, 3, 1.0), (2, 3, 1.0), (2, 1, 1.0), (3, 1, 1.0)]);
    let (_, paths) = max_flow(&cap, S, T);
    assert!((value(&paths) - 2.0).abs() < 1e-5, "B6: maximum flow {}, expected 2", value(&paths));
    // The lecture's "needs a back-edge" graph: only possible by undoing flow on a → b.
    let cap = graph(8, &[(0, 2, 4.0), (2, 3, 4.0), (3, 1, 4.0), (0, 4, 4.0), (4, 5, 4.0), (5, 3, 4.0), (2, 6, 4.0), (6, 7, 4.0), (7, 1, 4.0)]);
    let (flow, paths) = max_flow(&cap, S, T);
    assert!((value(&paths) - 8.0).abs() < 1e-5, "\"needs a back-edge\": flow {}, expected 8", value(&paths));
    assert_eq!(flow[2][3], 0.0, "\"needs a back-edge\": the flow on a → b must be undone");
}

#[test]
fn c4_min_cut_b4() {
    let cap = b1();
    let (flow, _) = max_flow(&cap, S, T);
    let side = source_side(&cap, &flow, S);
    let nodes: Vec<usize> = (0..6).filter(|&u| side[u]).collect();
    assert_eq!(nodes, vec![S, 2, 3], "B4: the nodes reachable from s are s, a, b");
    assert!((cut_cost(&cap, &side) - 9.0).abs() < 1e-5, "B4: the cut costs 9 = the flow");
}

#[test]
fn c4_flow_equals_cut_lecture_example() {
    // the lecture's example: flow 19, cut {s, b}
    let cap = graph(6, &[(0, 2, 10.0), (0, 3, 10.0), (2, 3, 2.0), (2, 4, 4.0), (2, 5, 8.0), (3, 5, 9.0), (5, 4, 6.0), (4, 1, 10.0), (5, 1, 10.0)]);
    let (flow, paths) = max_flow(&cap, S, T);
    let side = source_side(&cap, &flow, S);
    assert!((value(&paths) - 19.0).abs() < 1e-5, "lecture example: flow {}, expected 19", value(&paths));
    assert_eq!(side, vec![true, false, false, true, false, false], "lecture example: source side {{s, b}}");
}

#[test]
fn c4_flow_equals_cut_random_graphs() {
    let mut rng = Rng::new(8);
    for _ in 0..20 {
        let n = 8;
        let cap: Vec<Vec<f32>> = (0..n)
            .map(|u| (0..n).map(|v| if u != v && rng.uniform() < 0.4 { rng.below(10) as f32 } else { 0.0 }).collect())
            .collect();
        let (flow, paths) = max_flow(&cap, S, T);
        for u in 0..n {
            assert!((0..n).all(|v| flow[u][v] >= 0.0 && flow[u][v] <= cap[u][v] + 1e-4), "a flow above the capacity, or negative, at node {u}");
            if u != S && u != T {
                let (inflow, outflow): (f32, f32) = ((0..n).map(|v| flow[v][u]).sum(), (0..n).map(|v| flow[u][v]).sum());
                assert!((inflow - outflow).abs() < 1e-3, "node {u}: flow in {inflow} ≠ flow out {outflow}");
            }
        }
        let cut = cut_cost(&cap, &source_side(&cap, &flow, S));
        assert!((cut - value(&paths)).abs() < 1e-3, "flow {} but cut {cut}", value(&paths));
    }
}
