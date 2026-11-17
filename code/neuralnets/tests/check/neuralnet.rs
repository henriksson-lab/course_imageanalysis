// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p neuralnets neuralnet::

use neuralnets::neuralnet::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_neuron_b9() {
    let (w, b) = ([2.0, 1.0], -4.0);
    let points = [[1.0, 1.0], [2.0, 1.0], [0.0, 3.0], [3.0, 0.0]];
    let expected = [-1.0, 1.0, -1.0, 2.0];
    for (x, e) in points.iter().zip(expected) {
        let s = weighted_sum(&w, b, x);
        assert!(close(s, e, 1e-5), "B9a, point {x:?}: got {s}, expected {e}");
    }
    let p = sigmoid(weighted_sum(&w, b, &[1.0, 1.0]));
    assert!(close(p, 0.269, 1e-3), "B9c, sigmoid at (1, 1): got {p}, expected 0.269");
    let p = sigmoid(weighted_sum(&[4.0, 2.0], -8.0, &[2.0, 1.0]));
    assert!(close(p, 0.881, 1e-3), "B9c, doubled weights at (2, 1): got {p}, expected 0.881");
    assert!(relu(-3.0) == 0.0 && relu(2.5) == 2.5, "relu(-3) should be 0 and relu(2.5) should be 2.5");
}

/// The network of Part A: A, b, C, d.
fn part_a() -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, f32) {
    (vec![vec![2.0, -1.0], vec![1.0, -1.0], vec![-1.0, 1.0]], vec![0.0, 1.0, 0.0], vec![1.0, -1.0, 2.0], 1.0)
}

#[test]
fn c2_layer_b1() {
    let (a, b, _, _) = part_a();
    let h = layer(&a, &b, &[2.0, 1.0], false);
    assert!(h.len() == 3 && close(h[0], 3.0, 1e-5) && close(h[1], 2.0, 1e-5) && close(h[2], -1.0, 1e-5), "B1, A x + b: got {h:?}, expected [3, 2, -1]");
    let h = layer(&a, &b, &[2.0, 1.0], true);
    assert!(close(h[2], 0.0, 1e-6), "B1, after the ReLU: got {h:?}, expected [3, 2, 0]");
}

#[test]
fn c2_network_b1_b3() {
    let (a, b, c, d) = part_a();
    for (x, bend, e, what) in [
        ([2.0, 1.0], true, 2.0, "B1 with the ReLU"),
        ([2.0, 1.0], false, 0.0, "B1 without the ReLU"),
        ([1.0, 2.0], true, 3.0, "B3 with the ReLU"),
        ([1.0, 2.0], false, 3.0, "B3 without the ReLU"),
    ] {
        let out = network(&a, &b, &c, d, &x, bend);
        assert!(close(out, e, 1e-5), "{what}: got {out}, expected {e}");
    }
}

#[test]
fn c2_network_lecture() {
    // The slide "One input through the network, by hand": 6 with the bend, 4 without.
    let a = vec![vec![1.0, 1.0], vec![1.0, -1.0], vec![-1.0, 2.0]];
    let (b, c) = (vec![0.0; 3], vec![1.0, 2.0, 1.0]);
    let out = network(&a, &b, &c, 0.0, &[1.0, 2.0], true);
    assert!(close(out, 6.0, 1e-5), "with the bend: got {out}, expected 6");
    let out = network(&a, &b, &c, 0.0, &[1.0, 2.0], false);
    assert!(close(out, 4.0, 1e-5), "without the bend: got {out}, expected 4");
}

#[test]
fn c2_xor_b11() {
    let a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let (b, c) = (vec![0.0, -1.0], vec![1.0, -2.0]);
    for (x, e) in [([0.0, 0.0], 0.0), ([1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([1.0, 1.0], 0.0)] {
        let out = network(&a, &b, &c, 0.0, &x, true);
        assert!(close(out, e, 1e-5), "B11, corner {x:?}: got {out}, expected {e}");
    }
}

#[test]
fn c3_softmax_b12() {
    for z in [[2.0, 1.0, 0.0], [102.0, 101.0, 100.0]] {
        let p = softmax(&z);
        let expected = [0.665, 0.245, 0.090];
        for k in 0..3 {
            assert!(close(p[k], expected[k], 1e-3), "B12, scores {z:?}: got {p:?}, expected {expected:?}");
        }
    }
    let p = softmax(&[0.0, 1.5]);
    assert!(close(p[1], sigmoid(1.5), 1e-5), "B12c: two classes (0, s) should give sigmoid(s) = {}, got {}", sigmoid(1.5), p[1]);
}

#[test]
fn c4_train_neuron_separates_two_groups() {
    // Points left of x1 = 1 are class 0, right of it class 1.
    let points: Vec<Vec<f32>> = (0..20).map(|i| vec![(i % 10) as f32 * 0.2, (i / 10) as f32]).collect();
    let labels: Vec<f32> = points.iter().map(|p| if p[0] > 1.0 { 1.0 } else { 0.0 }).collect();
    let before = neuron_loss(&[0.0, 0.0], 0.0, &points, &labels);
    let (w, b) = train_neuron(&points, &labels, 1.0, 2000);
    let after = neuron_loss(&w, b, &points, &labels);
    assert!(after < 0.15 && after < before, "the loss should fall from {before:.3} to below 0.15, got {after:.3}");
    for (x, &y) in points.iter().zip(&labels) {
        let p = sigmoid(weighted_sum(&w, b, x));
        assert!((p > 0.5) == (y > 0.5), "point {x:?} (class {y}) is on the wrong side: p = {p:.3}");
    }
}

#[test]
fn c4_train_neuron_cannot_do_xor() {
    // Demo question P2: one neuron on XOR stays at the loss ln 2 = 0.693, a coin toss.
    let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let labels = [0.0, 1.0, 1.0, 0.0];
    let (w, b) = train_neuron(&points, &labels, 1.0, 2000);
    let loss = neuron_loss(&w, b, &points, &labels);
    assert!(close(loss, 2f32.ln(), 0.01), "XOR: got the loss {loss:.4}, expected ln 2 = 0.693");
}
