// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p diffusion drifting::

use diffusion::drifting::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_kernel_mean_b1() {
    let real = [0.0, 2.0];
    for (tau, want) in [(1.0, 0.538), (0.25, 0.036), (1000.0, 1.0)] {
        let m = kernel_mean(0.5, &real, tau, None);
        assert!(close(m, want, 0.002), "B1: kernel mean of {{0, 2}} seen from 0.5 with τ = {tau}: got {m:.3}, expected {want}");
    }
    let m = kernel_mean(0.5, &real, 0.01, None);
    assert!(close(m, 0.0, 1e-4), "τ = 0.01: the weights must not underflow; got {m}, expected the nearest value 0");
    let m = kernel_mean(0.5, &real, 1.0, Some(0));
    assert!(close(m, 2.0, 1e-6), "skipping entry 0 leaves only 2; got {m}");
    let m = kernel_mean(0.5, &[3.0], 1.0, Some(0));
    assert!(close(m, 0.5, 1e-6), "nothing left: expected x itself (0.5), got {m}");
}

#[test]
fn c2_field_b2() {
    let real = [0.0, 2.0];
    let v = field(&[0.5, 1.5], &real, 1.0, 1.0);
    assert!(close(v[0], -0.962, 0.002) && close(v[1], 0.962, 0.002), "B2a: field at {{0.5, 1.5}}: got {v:?}, expected [-0.962, 0.962]");
    let v = field(&[-1.0, 3.0], &real, 1.0, 1.0);
    assert!(close(v[0], -2.762, 0.002) && close(v[1], 2.762, 0.002), "B2c: field at {{-1, 3}}: got {v:?}, expected [-2.762, 2.762]");
    let v = field(&[0.0, 2.0], &real, 1.0, 1.0);
    assert!(close(v[0], -1.762, 0.002), "B2d: two points on the two data values: got {v:?}, expected V(0) = -1.762");
    let v = field(&[0.5, 1.5], &real, 1.0, 0.0);
    assert!(close(v[0], 0.038, 0.002), "attraction only (repel = 0) is V⁺: got {:.3}, expected 0.038 (B1)", v[0]);
}

#[test]
fn c3_coverage_b4() {
    let real = [0.0, 1.0, 2.0, 3.0, 10.0];
    let c = coverage(&real, &[2.5, 6.0], 1);
    assert!(close(c, 0.6, 1e-6), "B4a: got coverage {c}, expected 0.6 (2, 3 and 10 are covered)");
    let c = coverage(&real, &[2.5], 1);
    assert!(close(c, 0.4, 1e-6), "B4b, without the point at 6: got {c}, expected 0.4");
    let c = coverage(&real, &real, 1);
    assert!(close(c, 1.0, 1e-6), "the real values cover themselves: got {c}");
}

#[test]
fn c4_drift_step_b2() {
    let mut p = [0.5, 1.5];
    let norm = drift_step(&mut p, &[0.0, 2.0], 1.0, 0.5, 1.0);
    assert!(close(norm, 0.962, 0.002), "B2b: mean |V| before the step: got {norm:.3}, expected 0.962");
    assert!(close(p[0], 0.019, 0.002) && close(p[1], 1.981, 0.002), "B2b: after one step with η = 0.5: got {p:?}, expected [0.019, 1.981]");
    // many points on a single data value end up around it
    let mut p: Vec<f32> = (0..20).map(|i| -1.0 + 0.1 * i as f32).collect();
    let real: Vec<f32> = (0..200).map(|i| 0.01 * (i as f32 - 100.0)).collect(); // evenly from -1 to 1
    for _ in 0..200 {
        drift_step(&mut p, &real, 0.2, 0.5, 1.0);
    }
    let mean = p.iter().sum::<f32>() / p.len() as f32;
    assert!(close(mean, 0.0, 0.05), "20 points drifting onto evenly spread data on [-1, 1]: mean {mean:.3}, expected about 0");
    assert!(p.iter().all(|x| x.abs() < 1.2), "20 points drifting onto data on [-1, 1]: some ended far outside, {p:?}");
}

#[test]
fn c5_train_step_is_regression_on_fixed_targets() {
    let mut rng = Rng::new(3);
    let net = Net::new(4, &mut rng);
    let eps = [-1.0, -0.3, 0.4, 1.2];
    let real = [-1.5, -1.4, 1.5, 1.6];
    let (tau, lr) = (0.5, 0.01);
    // the targets f(ε) + V, frozen (stop-gradient)
    let f0: Vec<f32> = eps.iter().map(|&e| net.forward(e)).collect();
    let v = field(&f0, &real, tau, 1.0);
    let want_loss = v.iter().map(|x| x * x).sum::<f32>() / 4.0;
    let target: Vec<f32> = f0.iter().zip(&v).map(|(f, vi)| f + vi).collect();
    let loss_at = |n: &Net| eps.iter().zip(&target).map(|(&e, t)| (n.forward(e) - t).powi(2)).sum::<f32>() / 4.0;
    let mut trained = net.clone();
    let loss = train_step(&mut trained, &eps, &real, tau, lr, 1.0);
    assert!(close(loss, want_loss, 1e-5), "the returned loss must be the mean of V²: got {loss}, expected {want_loss}");
    // every weight must have moved by −lr × (finite-difference gradient of the frozen-target loss)
    let h = 1e-3;
    for k in 0..4 {
        let mut plus = net.clone();
        plus.w1[k] += h;
        let mut minus = net.clone();
        minus.w1[k] -= h;
        let g = (loss_at(&plus) - loss_at(&minus)) / (2.0 * h);
        let moved = trained.w1[k] - net.w1[k];
        assert!(close(moved, -lr * g, 2e-4), "w1[{k}] moved by {moved:.6}, expected −lr · gradient = {:.6}", -lr * g);
    }
    assert!(loss_at(&trained) < loss_at(&net), "one small step must lower the loss on the frozen targets");
}
