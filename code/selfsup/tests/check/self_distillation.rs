// Checks for the coding exercises of "Self-distillation and foundation models".
// Run: cargo test -p selfsup self_distillation::

use selfsup::self_distillation::*;

#[test]
fn c1_softmax_t_b2() {
    // B2: scores (2, 1, 0); centre (1, 0, 0) makes them (1, 1, 0)
    let p = softmax_t(&[2.0, 1.0, 0.0], &[0.0; 3], 1.0);
    assert!((p[0] - 0.665).abs() < 1e-3 && (p[1] - 0.245).abs() < 1e-3, "plain softmax of (2, 1, 0): (0.665, 0.245, 0.090), got {p:?}");
    let p = softmax_t(&[2.0, 1.0, 0.0], &[0.0; 3], 0.5);
    assert!((p[0] - 0.867).abs() < 1e-3, "sharpened (tau 0.5): 0.867 for the first, got {p:?}");
    let p = softmax_t(&[2.0, 1.0, 0.0], &[1.0, 0.0, 0.0], 1.0);
    assert!((p[0] - p[1]).abs() < 1e-6 && (p[0] - 0.422).abs() < 1e-3, "centred: (0.422, 0.422, 0.155), got {p:?}");
    let p = softmax_t(&[1000.0, 999.0], &[0.0; 2], 1.0);
    assert!(p.iter().all(|v| v.is_finite()), "large scores must not overflow: subtract the largest first");
}

#[test]
fn c2_distill_loss_and_gradient() {
    let t = [0.7, 0.2, 0.1];
    let z = [0.3, -0.1, 0.5];
    let (loss, grad) = distill_loss(&t, &z, 0.5);
    let h = 1e-3;
    for k in 0..3 {
        let (mut zp, mut zm) = (z, z);
        zp[k] += h;
        zm[k] -= h;
        let num = (distill_loss(&t, &zp, 0.5).0 - distill_loss(&t, &zm, 0.5).0) / (2.0 * h);
        assert!((num - grad[k]).abs() < 1e-3, "gradient {k}: formula {}, finite difference {num}", grad[k]);
    }
    let s = softmax_t(&z, &[0.0; 3], 0.5);
    let expected: f32 = -(0.7 * s[0].ln() + 0.2 * s[1].ln() + 0.1 * s[2].ln());
    assert!((loss - expected).abs() < 1e-5, "loss −Σ t ln s: expected {expected}, got {loss}");
    // the student already agrees: the gradient is 0
    let (_, g) = distill_loss(&s, &z, 0.5);
    assert!(g.iter().all(|v| v.abs() < 1e-6), "s = t: zero gradient, got {g:?}");
}

#[test]
fn c3_ema_b4() {
    let mut teacher = [0.0f32];
    for _ in 0..100 {
        ema(&mut teacher, &[1.0], 0.99);
    }
    assert!((teacher[0] - 0.634).abs() < 1e-3, "B4: after 100 steps with m = 0.99 the teacher is at 1 − 0.99^100 = 0.634, got {}", teacher[0]);
}

#[test]
fn c4_entropy() {
    assert!((entropy(&[0.25; 4]) - 4f32.ln()).abs() < 1e-6, "uniform over 4: ln 4");
    assert_eq!(entropy(&[1.0, 0.0, 0.0]), 0.0, "one-hot: 0 (and 0 ln 0 counts 0)");
}
