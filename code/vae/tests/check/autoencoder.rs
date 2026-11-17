// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p vae autoencoder::

use vae::autoencoder::*;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn close_vec(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

#[test]
fn c1_encode_decode_b1() {
    let (mean, u) = (vec![0.0, 0.0], vec![vec![0.8, 0.6]]);
    let s = pca_encode(&[3.0, 1.0], &mean, &u);
    assert!(close_vec(&s, &[3.0]), "B1: score of (3, 1): got {s:?}, expected [3.0]");
    let x = pca_decode(&s, &mean, &u);
    assert!(close_vec(&x, &[2.4, 1.8]), "B1: decoded: got {x:?}, expected [2.4, 1.8]");
}

#[test]
fn c1_on_the_line_and_perpendicular_b2() {
    let (mean, u) = (vec![0.0, 0.0], vec![vec![0.8, 0.6]]);
    let y = pca_decode(&pca_encode(&[-1.6, -1.2], &mean, &u), &mean, &u);
    assert!(close_vec(&y, &[-1.6, -1.2]), "B2: y lies on the line, decoded: got {y:?}, expected [-1.6, -1.2]");
    let w = pca_decode(&pca_encode(&[-0.6, 0.8], &mean, &u), &mean, &u);
    assert!(close_vec(&w, &[0.0, 0.0]), "B2: w is perpendicular, decoded: got {w:?}, expected the mean [0, 0]");
}

#[test]
fn c1_two_components_b3() {
    let (mean, u) = (vec![0.0, 0.0], vec![vec![0.8, 0.6], vec![-0.6, 0.8]]);
    let s = pca_encode(&[3.0, 1.0], &mean, &u);
    assert!(close_vec(&s, &[3.0, -1.0]), "B3: scores: got {s:?}, expected [3, -1]");
    let x = pca_decode(&s, &mean, &u);
    assert!(close_vec(&x, &[3.0, 1.0]), "B3: two components rebuild exactly: got {x:?}, expected [3, 1]");
}

#[test]
fn c2_error_b1_b5() {
    let e = reconstruction_error(&[vec![3.0, 1.0]], &[0.0, 0.0], &[vec![0.8, 0.6]]);
    assert!(close(e, 1.0), "B1: squared error: got {e}, expected 1.0");
    let r = std::f32::consts::FRAC_1_SQRT_2;
    let e = reconstruction_error(&[vec![0.8, 0.6]], &[0.5, 0.5], &[vec![r, r]]);
    assert!(close(e, 0.02), "B5: squared error with mean (0.5, 0.5): got {e}, expected 0.02");
    let s2 = pca_encode(&[0.8, 0.6], &[0.5, 0.5], &[vec![r, -r]]);
    assert!(close(s2[0] * s2[0], e), "B5: the lost error should be the PC2 score squared, {} vs {e}", s2[0] * s2[0]);
}

#[test]
fn c2_error_is_trace_minus_variance_b9() {
    // Four centred points with covariance S = [[2, 1], [1, 2]] (divide by n): ±√6 u1 and ±√2 u2,
    // with u1 = (1, 1)/√2 (eigenvalue 3) and u2 = (1, −1)/√2 (eigenvalue 1).
    let r = std::f32::consts::FRAC_1_SQRT_2;
    let (a, b) = (6.0f32.sqrt() * r, 2.0f32.sqrt() * r);
    let pts = vec![vec![a, a], vec![-a, -a], vec![b, -b], vec![-b, b]];
    let e = reconstruction_error(&pts, &[0.0, 0.0], &[vec![1.0, 0.0]]);
    assert!(close(e, 2.0), "B9, u = (1, 0): got {e}, expected tr S − uᵀSu = 4 − 2 = 2");
    let e = reconstruction_error(&pts, &[0.0, 0.0], &[vec![r, r]]);
    assert!(close(e, 1.0), "B9, u = (1, 1)/√2: got {e}, expected 4 − 3 = 1");
}

#[test]
fn c3_dense_and_forward() {
    let layer = Layer { w: vec![vec![1.0, -2.0], vec![0.5, 0.5]], b: vec![0.5, -1.0] };
    let out = dense(&layer, &[1.0, 1.0], false);
    assert!(close_vec(&out, &[-0.5, 0.0]), "w·(1, 1) + b: got {out:?}, expected [-0.5, 0.0]");
    let out = dense(&layer, &[3.0, 1.0], true);
    assert!(close_vec(&out, &[1.5, 1.0]), "ReLU(w·(3, 1) + b): got {out:?}, expected [1.5, 1.0]");
    let out = dense(&layer, &[0.0, 1.0], true);
    assert!(close_vec(&out, &[0.0, 0.0]), "ReLU of negative values: got {out:?}, expected [0, 0]");
    let ae = Autoencoder::new(16, 1);
    let outs = forward(&ae, &[0.5, 0.2]);
    let sizes: Vec<usize> = outs.iter().map(|o| o.len()).collect();
    assert_eq!(sizes, vec![16, 1, 16, 2], "forward should return h1, z, h2, x̂ with 16, 1, 16, 2 numbers");
    assert!(outs[0].iter().chain(&outs[2]).all(|&v| v >= 0.0), "the hidden layers come after a ReLU: no negative values");
}

#[test]
fn c4_gradient_matches_finite_differences() {
    // Change one weight a little and compare the change of the error with the gradient step.
    let data = moving_spot(20, 0.0, 3);
    let ae = Autoencoder::new(16, 2);
    let error = |a: &Autoencoder| {
        let mut probe = a.clone();
        train_step(&mut probe, &data, 0.0)
    };
    let mut stepped = ae.clone();
    let rate = 1e-3;
    train_step(&mut stepped, &data, rate);
    for (k, o, i) in [(0, 3, 1), (1, 0, 5), (2, 7, 0), (3, 1, 9)] {
        let grad = (ae.layers[k].w[o][i] - stepped.layers[k].w[o][i]) / rate;
        let h = 1e-2;
        let mut plus = ae.clone();
        plus.layers[k].w[o][i] += h;
        let mut minus = ae.clone();
        minus.layers[k].w[o][i] -= h;
        let numeric = (error(&plus) - error(&minus)) / (2.0 * h);
        assert!((grad - numeric).abs() < 1e-3 + 0.05 * numeric.abs(), "layer {k}, weight [{o}][{i}]: gradient {grad}, but the error changes at the rate {numeric}");
    }
}

#[test]
fn c4_training_beats_pca_on_the_moving_spot() {
    let data = moving_spot(100, 0.02, 7);
    let mut ae = Autoencoder::new(16, 1);
    let first = train_step(&mut ae, &data, 0.05);
    let mut last = first;
    for _ in 0..3000 {
        last = train_step(&mut ae, &data, 0.05);
    }
    assert!(last < 0.3 * first, "the error should fall during training: from {first} to {last}");
    assert!(last < 0.01, "after 3000 steps the error should be below PCA's (about 0.017), got {last}");
}
