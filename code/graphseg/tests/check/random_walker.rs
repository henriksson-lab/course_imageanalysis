// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Nodes of a chain are numbered from 1 in the exercises and from 0 here.
// Run: cargo test -p graphseg random_walker::

use imgcourse::{Image, Rng};
use graphseg::random_walker::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn assert_values(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    for k in 0..expected.len() {
        assert!(close(got[k], expected[k], tol), "{what}: got {got:.3?}, expected {expected:.3?}");
    }
}

/// A chain as a one-row image: given weights, seeds 1 (object) and 2 (background).
fn chain(weights: &[f32], seeds: &[u32]) -> (Seeds, Weights) {
    let n = seeds.len();
    let s = Image::from_vec(n, 1, seeds.to_vec());
    let wx = Image::from_fn(n, 1, |x, _| if x + 1 < n { weights[x] } else { 0.0 });
    (s, Weights { wx, wy: Image::new(n, 1, 0.0) })
}

#[test]
fn c1_chain_b1() {
    let v = chain_voltages(&[1.0, 1.0, 0.25, 1.0], &[Some(1.0), None, None, None, Some(0.0)]);
    assert_values(&v, &[1.0, 6.0 / 7.0, 5.0 / 7.0, 1.0 / 7.0, 0.0], 1e-5, "B1(a), voltages");
}

#[test]
fn c1_chain_b3_b4() {
    let w = [0.2, 1.0, 1.0, 1.0, 1.0, 0.25];
    let v = chain_voltages(&w, &[Some(1.0), None, None, None, None, None, Some(0.0)]);
    let expected = [13.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0].map(|k| k / 13.0);
    assert_values(&v, &expected, 1e-5, "B3(a), seeds at the ends");
    let v = chain_voltages(&w, &[None, None, Some(1.0), None, None, None, Some(0.0)]);
    let expected = [1.0, 1.0, 1.0, 6.0 / 7.0, 5.0 / 7.0, 4.0 / 7.0, 0.0];
    assert_values(&v, &expected, 1e-5, "B4, object seed moved to node 3");
}

#[test]
fn c2_weights() {
    let img = Image::from_rows(&[[0.5, 0.5, 0.6], [0.7, 0.5, 0.5]]);
    let w = edge_weights(&img, 0.1);
    assert!(close(w.wx[(0, 0)], 1.0, 1e-6), "equal neighbours: got {}, expected 1", w.wx[(0, 0)]);
    let e = (-0.5f32).exp();
    assert!(close(w.wx[(1, 0)], e, 1e-4), "difference σ: got {}, expected e^(−1/2) = {e:.4}", w.wx[(1, 0)]);
    assert!(close(w.wy[(0, 0)], (-2.0f32).exp(), 1e-4), "difference 2σ downwards: got {}, expected e^(−2)", w.wy[(0, 0)]);
    assert!(w.wx[(2, 0)] == 0.0 && w.wy[(0, 1)] == 0.0, "no neighbour beyond the last column / row: weight 0");
    // B6(b): adding a constant changes nothing
    let w2 = edge_weights(&img.map(|v| v + 0.3), 0.1);
    assert!(close(w2.wx[(1, 0)], w.wx[(1, 0)], 1e-5), "B6(b): weights depend only on differences");
}

/// The 3 × 3 grid of B5: weights 1, object seed top left, background seed bottom right.
fn b5() -> (Seeds, Weights) {
    let mut s = Image::new(3, 3, 0u32);
    s[(0, 0)] = 1;
    s[(2, 2)] = 2;
    let wx = Image::from_fn(3, 3, |x, _| if x < 2 { 1.0 } else { 0.0 });
    let wy = Image::from_fn(3, 3, |_, y| if y < 2 { 1.0 } else { 0.0 });
    (s, Weights { wx, wy })
}

const B5: [f32; 9] = [1.0, 2.0 / 3.0, 0.5, 2.0 / 3.0, 0.5, 1.0 / 3.0, 0.5, 1.0 / 3.0, 0.0];

#[test]
fn c3_averaging_b5() {
    let (s, w) = b5();
    let mut x = s.map(|v| if v == 1 { 1.0 } else { 0.0 });
    let mut sweeps = 0;
    while averaging_sweep(&mut x, &s, &w) > 1e-6 {
        sweeps += 1;
        assert!(sweeps < 1000, "averaging did not converge in 1000 sweeps");
    }
    assert_values(x.pixels(), &B5, 1e-4, "B5 after averaging (row by row)");
    assert!(x[(0, 0)] == 1.0 && x[(2, 2)] == 0.0, "seeds must not change");
}

#[test]
fn c4_walker_b5() {
    let (s, w) = b5();
    let (x, _) = random_walker(&s, &w, 1, 1e-6, 100);
    assert_values(x.pixels(), &B5, 1e-4, "B5, random walker");
}

#[test]
fn c4_walker_b1_b2() {
    let (s, w) = chain(&[1.0, 1.0, 0.25, 1.0], &[1, 0, 0, 0, 2]);
    let (x, _) = random_walker(&s, &w, 1, 1e-6, 100);
    assert_values(x.pixels(), &[1.0, 6.0 / 7.0, 5.0 / 7.0, 1.0 / 7.0, 0.0], 1e-4, "B1 as a one-row image");
    // B2: free node (1, 0) joined to an object seed (w = 3) and two background seeds (w = 1)
    let s = Image::from_rows(&[[1u32, 0, 2], [2, 2, 2]]);
    let wx = Image::from_rows(&[[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let wy = Image::from_rows(&[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let (x, _) = random_walker(&s, &Weights { wx, wy }, 1, 1e-6, 100);
    assert!(close(x[(1, 0)], 0.6, 1e-4), "B2(a): got {}, expected 3/5", x[(1, 0)]);
}

#[test]
fn c4_three_labels_sum_to_one() {
    let img = Image::from_fn(12, 10, |x, y| ((x * 7 + y * 3) % 5) as f32 / 5.0);
    let w = edge_weights(&img, 0.3);
    let mut s = Image::new(12, 10, 0u32);
    s[(1, 1)] = 1;
    s[(10, 2)] = 2;
    s[(5, 8)] = 3;
    let probs: Vec<Image<f32>> = (1..=3).map(|k| random_walker(&s, &w, k, 1e-6, 1000).0).collect();
    for (x, y) in s.coords() {
        let sum: f32 = probs.iter().map(|p| p[(x, y)]).sum();
        assert!(close(sum, 1.0, 1e-3), "pixel ({x}, {y}): the three probabilities sum to {sum}, expected 1");
        for p in &probs {
            assert!((-1e-4..=1.0001).contains(&p[(x, y)]), "pixel ({x}, {y}): probability {} outside [0, 1]", p[(x, y)]);
        }
    }
}

#[test]
fn c5_walks_b1() {
    let (s, w) = chain(&[1.0, 1.0, 0.25, 1.0], &[1, 0, 0, 0, 2]);
    let p = walk_estimate(&s, &w, 1, (2, 0), 4000, &mut Rng::new(1));
    assert!(close(p, 5.0 / 7.0, 0.03), "4000 walks from node 3: got {p:.3}, expected about 5/7 = 0.714 (± 0.03)");
}

#[test]
fn c4a_laplacian_of_the_free_pixels() {
    // B2: the free node has weights 3 + 1 + 1 = 5; all its neighbours are seeds
    let s = Image::from_rows(&[[1u32, 0, 2], [2, 2, 2]]);
    let wx = Image::from_rows(&[[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let wy = Image::from_rows(&[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let v = Image::from_rows(&[[7.0, 2.0, 7.0], [7.0, 7.0, 7.0]]);
    let out = apply_lu(&v, &s, &Weights { wx, wy });
    assert!(close(out[(1, 0)], 10.0, 1e-5), "B2 node: got {}, expected d·v = 5 · 2 = 10 (seed neighbours do not count)", out[(1, 0)]);
    assert!(out[(0, 0)] == 0.0, "seed pixels give 0, got {}", out[(0, 0)]);
    // a chain of three free nodes between two seeds, weights 1: rows (2 −1 0), (−1 2 −1), (0 −1 2)
    let (s, w) = chain(&[1.0, 1.0, 1.0, 1.0], &[1, 0, 0, 0, 2]);
    let out = apply_lu(&Image::from_vec(5, 1, vec![0.0, 1.0, 2.0, 3.0, 0.0]), &s, &w);
    assert_values(out.pixels(), &[0.0, 0.0, 0.0, 4.0, 0.0], 1e-5, "free chain (1, 2, 3)");
}
