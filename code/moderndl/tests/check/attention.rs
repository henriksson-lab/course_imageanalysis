// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p moderndl attention::

use moderndl::attention::*;
use imgcourse::{samples, Image, Rng};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn b1_keys() -> Vec<Vec<f32>> {
    vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]
}

fn b1_values() -> Vec<Vec<f32>> {
    vec![vec![10.0], vec![0.0], vec![4.0]]
}

#[test]
fn r1_softmax_average_b1a() {
    let (w, out) = weights_and_average(&[1.0, 0.0, 1.0], &b1_values());
    let expected = [0.422, 0.155, 0.422];
    assert!(w.iter().zip(expected).all(|(a, b)| close(*a, b, 1e-3)), "B1 (a): weights for scores 1, 0, 1: got {w:?}, expected {expected:?}");
    assert!(close(out[0], 5.91, 0.01), "B1 (a): output: got {}, expected 5.91", out[0]);
    let (w, _) = weights_and_average(&[1000.0, 0.0, 1000.0], &b1_values());
    assert!(close(w[0], 0.5, 1e-4), "very large scores should not overflow: got weights {w:?}, expected 0.5, 0, 0.5");
}

#[test]
fn r2_attend_temperatures_b1() {
    for (t, expected) in [(1.0, 5.91), (0.5, 6.56), (2.0, 5.37), (0.01, 7.0), (1000.0, 4.67)] {
        let (w, out) = attend(&[1.0, 0.0], &b1_keys(), &b1_values(), t);
        assert!(close(w.iter().sum::<f32>(), 1.0, 1e-5), "the weights should sum to 1, got {w:?}");
        assert!(close(out[0], expected, 0.01), "B1: temperature {t}: got output {}, expected {expected}", out[0]);
    }
}

#[test]
fn r3_patch_attention_tiny_temperature_keeps_the_image() {
    let img = samples::cell().crop(64, 64, 32, 32);
    let out = patch_attention(&img, 8, 100.0, 1.0, 1e-4);
    for (x, y) in img.coords() {
        assert!(close(out[(x, y)], img[(x, y)], 1e-4), "temperature → 0: every patch keeps itself; pixel ({x}, {y}) got {}, expected {}", out[(x, y)], img[(x, y)]);
    }
}

#[test]
fn r3_patch_attention_without_position_only_reorders_b8() {
    // Content-only attention is permutation-equivariant: shuffling the patches shuffles the output.
    let img = samples::cell().crop(100, 60, 32, 32);
    let p = 8;
    let patches = to_patches(&img, p);
    let mut order: Vec<usize> = (0..patches.len()).collect();
    let mut rng = Rng::new(3);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let shuffled = from_patches(&order.iter().map(|&k| patches[k].clone()).collect::<Vec<_>>(), p, 32);
    let out = to_patches(&patch_attention(&img, p, 50.0, 0.0, 1.0), p);
    let out_shuffled = to_patches(&patch_attention(&shuffled, p, 50.0, 0.0, 1.0), p);
    for (i, &k) in order.iter().enumerate() {
        for d in 0..p * p {
            assert!(close(out_shuffled[i][d], out[k][d], 1e-4), "B8: patch {i} of the shuffled output should be patch {k} of the output");
        }
    }
    // With a position weight it is not: the grid distance depends on where a patch sits.
    let differs = to_patches(&patch_attention(&shuffled, p, 50.0, 1.0, 1.0), p);
    let with_pos = to_patches(&patch_attention(&img, p, 50.0, 1.0, 1.0), p);
    let largest = order.iter().enumerate().map(|(i, &k)| (0..p * p).map(|d| (differs[i][d] - with_pos[k][d]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max);
    assert!(largest > 0.01, "with a position weight, shuffling should change the result, largest difference {largest}");
}

#[test]
fn r3_position_only_on_a_flat_image() {
    let flat = Image::new(32, 32, 0.25f32);
    let out = patch_attention(&flat, 8, 0.0, 1.0, 1.0);
    assert!(out.pixels().iter().all(|&v| close(v, 0.25, 1e-5)), "averaging equal patches gives the same patches");
}

#[test]
fn r4_ece_b6() {
    let conf = [0.55, 0.60, 0.65, 0.70, 0.72, 0.80, 0.85, 0.90, 0.95, 0.99];
    let correct = [true, false, true, false, true, true, false, true, true, true];
    let got = ece(&conf, &correct, 2);
    assert!(close(got, 0.071, 1e-3), "B6: ECE with the bins [0.5, 0.75) and [0.75, 1]: got {got}, expected 0.071");
    let perfect = ece(&[0.75, 0.75, 0.75, 0.75], &[true, true, true, false], 5);
    assert!(close(perfect, 0.0, 1e-6), "confidence 0.75 and 3 of 4 right: calibrated, ECE 0, got {perfect}");
}

#[test]
fn r5_fit_temperature_b10() {
    assert!(close(sigmoid(2.0), 0.881, 1e-3) && close(sigmoid(2.0 / 2.0), 0.731, 1e-3), "B10: σ(2) = 0.88, σ(2/2) = 0.73");
    // Labels drawn from calibrated probabilities σ(z); the "network" reports 3z: T should be about 3.
    let mut rng = Rng::new(7);
    let (mut logits, mut labels) = (Vec::new(), Vec::new());
    for _ in 0..4000 {
        let z = 2.0 * rng.normal();
        labels.push(rng.uniform() < sigmoid(z));
        logits.push(3.0 * z);
    }
    let candidates: Vec<f32> = (1..=100).map(|k| 0.1 * k as f32).collect();
    let t = fit_temperature(&logits, &labels, &candidates);
    assert!(close(t, 3.0, 0.4), "overconfident logits (× 3): fitted T = {t}, expected about 3");
}
