// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p training optimisers::

use training::optimisers::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn c1_gd_step() {
    // B1: the valley 0.1 a^2 + 5 b^2 at (-1.8, 0.8): gradient (-0.36, 8)
    let mut w = vec![-1.8, 0.8];
    gd_step(&mut w, &[-0.36, 8.0], 0.1);
    assert!(close(w[0], -1.764, 1e-5) && close(w[1], 0.0, 1e-5), "B1, eta 0.1: got {w:?}, expected [-1.764, 0]");
    let mut w = vec![-1.8, 0.8];
    gd_step(&mut w, &[-0.36, 8.0], 0.3);
    assert!(close(w[1], -1.6, 1e-5), "B1, eta 0.3: b should jump to -1.6, got {}", w[1]);
}

#[test]
fn c2_momentum_constant_gradient_b2() {
    // B2: a constant gradient 1, beta 0.9: v = 1, 1.9, 2.71; w = -eta (1 + 1.9 + 2.71)
    let (mut w, mut v) = (vec![0.0], vec![0.0]);
    let mut vs = Vec::new();
    for _ in 0..3 {
        momentum_step(&mut w, &mut v, &[1.0], 0.1, 0.9);
        vs.push(v[0]);
    }
    assert!(close(vs[0], 1.0, 1e-5) && close(vs[1], 1.9, 1e-5) && close(vs[2], 2.71, 1e-5), "B2: velocities {vs:?}, expected [1, 1.9, 2.71]");
    assert!(close(w[0], -0.561, 1e-5), "B2: w after three steps {}, expected -0.561", w[0]);
    // the velocity approaches g / (1 - beta) = 10
    for _ in 0..200 {
        momentum_step(&mut w, &mut v, &[1.0], 0.1, 0.9);
    }
    assert!(close(v[0], 10.0, 1e-3), "B2: the velocity should approach 10, got {}", v[0]);
}

#[test]
fn c3_adam_first_step_is_eta_b3() {
    // B3: with bias correction the first step is eta in every weight, whatever the gradient's size
    let mut w = vec![0.0, 0.0, 0.0];
    let mut s = AdamState::new(3);
    adam_step(&mut w, &mut s, &[0.001, -5.0, 1000.0], 0.1);
    for (got, want) in w.iter().zip([-0.1, 0.1, -0.1]) {
        assert!(close(*got, want, 1e-4), "B3: first Adam step {w:?}, expected [-0.1, 0.1, -0.1]");
    }
    assert_eq!(s.t, 1, "the step counter should be 1 after one step");
}

#[test]
fn c3_adam_without_bias_correction_b3() {
    // B3(c): without the correction, m = 0.1 g and v = 0.001 g^2, so the step is 0.1/sqrt(0.001) = 3.16 eta
    let mut w = vec![0.0];
    let mut s = AdamState::new(1);
    s.bias_correction = false;
    adam_step(&mut w, &mut s, &[2.0], 0.1);
    assert!(close(w[0], -0.31623, 1e-3), "B3c: step without bias correction {}, expected -0.316", w[0]);
}

#[test]
fn c3_adam_sign_change_b4() {
    // B4: gradient +1, then -1: the second step is small and positive (+0.0526 eta)
    let mut w = vec![0.0];
    let mut s = AdamState::new(1);
    adam_step(&mut w, &mut s, &[1.0], 1.0);
    let after1 = w[0];
    adam_step(&mut w, &mut s, &[-1.0], 1.0);
    assert!(close(after1, -1.0, 1e-4), "B4: first step {after1}, expected -1");
    assert!(close(w[0] - after1, 0.05263, 1e-3), "B4: second step {}, expected +0.0526", w[0] - after1);
    assert!(close(s.m[0], -0.01, 1e-4), "B4: stored m should be the uncorrected running mean -0.01, got {}", s.m[0]);
}

#[test]
fn c4_learning_rate_schedule_b5() {
    // B5: warm-up 30 of 300 steps, eta_max 0.1
    for (t, want) in [(0, 0.1 / 30.0), (9, 0.1 / 3.0), (29, 0.1), (30, 0.1), (165, 0.05), (299, 0.0000034)] {
        let got = learning_rate(t, 300, 30, 0.1);
        assert!((got - want).abs() < 2e-5, "B5: learning rate at step {t}: got {got}, expected {want}");
    }
}
