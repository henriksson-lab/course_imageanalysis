// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day36 backprop::

use day36::backprop::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The network of B3: one hidden unit, w1 = 0.5, b1 = 0, w2 = 1.5, b2 = 0.
fn b3() -> Net {
    Net { w1: vec![0.5], b1: vec![0.0], w2: vec![1.5], b2: 0.0 }
}

fn assert_net(got: &Net, expected: [f32; 4], what: &str) {
    let g = [got.w1[0], got.b1[0], got.w2[0], got.b2];
    for i in 0..4 {
        assert!(close(g[i], expected[i], 1e-4), "{what}: got (w1, b1, w2, b2) = {g:?}, expected {expected:?}");
    }
}

#[test]
fn c1_cross_entropy_b1() {
    for (p, e) in [(0.9, 0.105), (0.5, 0.693), (0.01, 4.605)] {
        let l = cross_entropy(&[p, 1.0 - p], 0);
        assert!(close(l, e, 1e-3), "B1a, probability {p} for the true class: got {l}, expected {e}");
    }
    let p = [0.7, 0.2, 0.1];
    let l = cross_entropy(&p, 1);
    assert!(close(l, 1.609, 1e-3), "B1b, true class 2 of (0.7, 0.2, 0.1): got {l}, expected 1.609");
}

#[test]
fn c2_forward_b3a() {
    let f = forward(&b3(), 2.0);
    assert!(close(f.z[0], 1.0, 1e-6) && close(f.h[0], 1.0, 1e-6), "B3a: got z = {:?}, h = {:?}, expected 1 and 1", f.z, f.h);
    assert!(close(f.y_hat, 1.5, 1e-6), "B3a: got y_hat = {}, expected 1.5", f.y_hat);
    let f = forward(&b3(), -1.0);
    assert!(close(f.h[0], 0.0, 1e-6) && close(f.y_hat, 0.0, 1e-6), "B3d (x = -1): got h = {:?}, y_hat = {}, expected 0 and 0", f.h, f.y_hat);
}

#[test]
fn c3_backward_b3b() {
    let g = backward(&b3(), 2.0, 1.0);
    assert_net(&g, [3.0, 1.5, 1.0, 1.0], "B3b, the gradients");
}

#[test]
fn c3_backward_dead_relu_b3d() {
    let g = backward(&b3(), -1.0, 1.0);
    assert_net(&g, [0.0, 0.0, 0.0, -2.0], "B3d, the ReLU is off: only b2 gets a gradient");
}

#[test]
fn c4_train_step_b3c() {
    for (eta, w, after) in [(0.1, [0.2, -0.15, 1.4, -0.1], 0.5625), (0.05, [0.35, -0.075, 1.45, -0.05], 0.021)] {
        let mut net = b3();
        let before = train_step(&mut net, &[2.0], &[1.0], eta);
        assert!(close(before, 0.25, 1e-5), "B3c: train_step should return the loss before the step, 0.25; got {before}");
        assert_net(&net, w, &format!("B3c, the weights after one step with eta = {eta}"));
        let l = loss(&net, 2.0, 1.0);
        assert!(close(l, after, 1e-3), "B3c, eta = {eta}: got the new loss {l}, expected {after}");
    }
}

#[test]
fn c4_train_step_averages() {
    // Two examples: the step must use the mean gradient, so two copies of B3's example
    // give the same step as one.
    let mut one = b3();
    train_step(&mut one, &[2.0], &[1.0], 0.1);
    let mut two = b3();
    train_step(&mut two, &[2.0, 2.0], &[1.0, 1.0], 0.1);
    assert_net(&two, [one.w1[0], one.b1[0], one.w2[0], one.b2], "two copies of the same example");
}

#[test]
fn c5_numerical_gradient_agrees() {
    // Demo question R1: finite differences agree with backpropagation.
    let net = Net::new(4);
    for (x, y) in [(0.3, 0.8), (0.9, 0.1), (0.55, 0.5)] {
        let a = backward(&net, x, y).to_vec();
        let b = numerical_gradient(&net, x, y, 1e-3).to_vec();
        for i in 0..a.len() {
            assert!(close(a[i], b[i], 2e-3), "x = {x}: backprop gives {a:?}, finite differences {b:?}");
        }
    }
}
