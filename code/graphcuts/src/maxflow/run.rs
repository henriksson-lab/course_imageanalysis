// Runs maximum flow on the graphs of the lecture and the exercise: prints every augmenting
// path, the final flow and the minimum cut. Then the cost on larger grid graphs.
//   cargo run -p graphcuts -- maxflow                      (your code)
//   cargo run -p graphcuts --features solution -- maxflow  (the model solution)

use imgcourse::Rng;
use crate::maxflow::*;

/// A named graph: node names (s first, t second) and edges (u, v, capacity).
fn graph(names: &[&str], edges: &[(&str, &str, f32)]) -> Vec<Vec<f32>> {
    let id = |name: &str| names.iter().position(|n| *n == name).expect("unknown node");
    let mut cap = vec![vec![0.0; names.len()]; names.len()];
    for &(u, v, c) in edges {
        cap[id(u)][id(v)] = c;
    }
    cap
}

fn report(title: &str, names: &[&str], cap: &[Vec<f32>]) {
    println!("{title}");
    let (flow, paths) = max_flow(cap, 0, 1);
    let mut total = 0.0;
    for (path, b) in &paths {
        total += b;
        let p: Vec<&str> = path.iter().map(|&u| names[u]).collect();
        println!("  path {:<16} bottleneck {b:4}   flow so far {total}", p.join(" "));
    }
    let side = source_side(cap, &flow, 0);
    let s: Vec<&str> = (0..names.len()).filter(|&u| side[u]).map(|u| names[u]).collect();
    let mut cut_edges = Vec::new();
    for u in 0..names.len() {
        for v in 0..names.len() {
            if side[u] && !side[v] && cap[u][v] > 0.0 {
                cut_edges.push(format!("{}→{} ({}/{})", names[u], names[v], flow[u][v], cap[u][v]));
            }
        }
    }
    println!("  no path left: S = {{{}}}, cut edges {}, cut = {} = flow {total}\n", s.join(", "), cut_edges.join(", "), cut_cost(cap, &side));
}

pub fn main() {
    let names = ["s", "t", "a", "b", "c", "d"];
    let lecture = graph(&names, &[
        ("s", "a", 10.0), ("s", "b", 10.0), ("a", "b", 2.0), ("a", "c", 4.0), ("a", "d", 8.0),
        ("b", "d", 9.0), ("d", "c", 6.0), ("c", "t", 10.0), ("d", "t", 10.0),
    ]);
    report("the lecture's example", &names, &lecture);
    let exercise = graph(&names, &[
        ("s", "a", 8.0), ("s", "b", 4.0), ("a", "b", 8.0), ("a", "c", 5.0), ("b", "d", 4.0),
        ("c", "d", 7.0), ("c", "t", 4.0), ("d", "t", 7.0),
    ]);
    report("the graph of B1–B5", &names, &exercise);
    let names8 = ["s", "t", "a", "b", "c", "g", "d", "e"];
    let back = graph(&names8, &[
        ("s", "a", 4.0), ("a", "b", 4.0), ("b", "t", 4.0), ("s", "c", 4.0), ("c", "g", 4.0),
        ("g", "b", 4.0), ("a", "d", 4.0), ("d", "e", 4.0), ("e", "t", 4.0),
    ]);
    report("needs a back-edge", &names8, &back);

    // The cost: a k × k grid of nodes with random capacities to the four neighbours, the
    // left column joined to s and the right column to t (an image graph in miniature).
    println!("grid graphs (random capacities 1..9):   nodes   augmenting paths   time");
    for k in [10, 20, 30] {
        let n = k * k + 2;
        let mut rng = Rng::new(k as u64);
        let mut cap = vec![vec![0.0; n]; n];
        let node = |x: usize, y: usize| 2 + y * k + x;
        for y in 0..k {
            for x in 0..k {
                if x + 1 < k {
                    cap[node(x, y)][node(x + 1, y)] = 1.0 + rng.below(9) as f32;
                    cap[node(x + 1, y)][node(x, y)] = 1.0 + rng.below(9) as f32;
                }
                if y + 1 < k {
                    cap[node(x, y)][node(x, y + 1)] = 1.0 + rng.below(9) as f32;
                    cap[node(x, y + 1)][node(x, y)] = 1.0 + rng.below(9) as f32;
                }
            }
            cap[0][node(0, y)] = 10.0;
            cap[node(k - 1, y)][1] = 10.0;
        }
        let start = std::time::Instant::now();
        let (_, paths) = max_flow(&cap, 0, 1);
        println!("                                         {n:5}   {:10}         {:.3} s", paths.len(), start.elapsed().as_secs_f32());
    }
}
