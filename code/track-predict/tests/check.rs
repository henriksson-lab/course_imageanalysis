// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p track-predict

use imgcourse::Rng;
use track_predict::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_update_b1() {
    let pred = Estimate { x: 10.0, p: 4.0 };
    // B1(a): R = 1, 4, 16
    for (r, k, x, p) in [(1.0, 0.8, 12.4, 0.8), (4.0, 0.5, 11.5, 2.0), (16.0, 0.2, 10.6, 3.2)] {
        let (e, gain) = update(pred, 13.0, r);
        assert!(close(gain, k, 1e-5), "B1, R = {r}: gain {gain}, expected {k}");
        assert!(close(e.x, x, 1e-4), "B1, R = {r}: estimate {}, expected {x}", e.x);
        assert!(close(e.p, p, 1e-4), "B1, R = {r}: variance {}, expected {p}", e.p);
    }
}

#[test]
fn c1_predict_b2() {
    let e = predict(Estimate { x: 12.4, p: 0.8 }, 1.0);
    assert!(close(e.x, 12.4, 1e-5) && close(e.p, 1.8, 1e-5), "B2(a): got {e:?}, expected x 12.4, P 1.8");
}

#[test]
fn c2_missed_frame_b2() {
    // B2: from 12.4 ± 0.8 (Q = R = 1): a frame without a detection, then the measurement 15
    let out = run_filter(Estimate { x: 12.4, p: 0.8 }, &[None, Some(15.0)], 1.0, 1.0);
    assert!(close(out[0].x, 12.4, 1e-4) && close(out[0].p, 1.8, 1e-4), "B2(b): after the missed frame {:?}, expected x 12.4, P 1.8", out[0]);
    assert!(close(out[1].x, 14.32, 0.01), "B2(c): estimate {}, expected 14.32", out[1].x);
    assert!(close(out[1].p, 0.737, 0.001), "B2(c): P {}, expected 0.737 (the gain was 2.8/3.8)", out[1].p);
}

#[test]
fn c2_steady_state_golden_ratio_b8() {
    // Q = R: the gain settles at (√5 − 1)/2 = 0.618, and then P = K R.
    let zs = vec![Some(0.0); 60];
    let out = run_filter(Estimate { x: 0.0, p: 100.0 }, &zs, 1.0, 1.0);
    let k = out.last().unwrap().p; // P = K R with R = 1
    assert!(close(k, 0.618, 0.001), "B8: steady-state gain {k}, expected 0.618");
}

#[test]
fn c3_velocity_from_positions_b3() {
    let start = CvEstimate { x: [0.0, 0.0], p: [[1.0, 0.0], [0.0, 1.0]] };
    let pred = cv_predict(start, 1.0, 0.1);
    let expected = [[2.0, 1.0], [1.0, 1.1]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(close(pred.p[i][j], expected[i][j], 1e-5), "B3(a): P⁻ = {:?}, expected {expected:?}", pred.p);
        }
    }
    // B3(c): a measurement 3 units ahead of the prediction
    let (e, k) = cv_update(pred, pred.x[0] + 3.0, 1.0);
    assert!(close(k[0], 2.0 / 3.0, 1e-5) && close(k[1], 1.0 / 3.0, 1e-5), "B3(b): gain {k:?}, expected (2/3, 1/3)");
    assert!(close(e.x[0], 2.0, 1e-4) && close(e.x[1], 1.0, 1e-4), "B3(c): position {} and velocity {}, expected 2 and 1", e.x[0], e.x[1]);
    let p = [[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 23.0 / 30.0]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(close(e.p[i][j], p[i][j], 1e-5), "B7: P = {:?}, expected {p:?}", e.p);
        }
    }
}

#[test]
fn c3_locks_on_to_the_velocity() {
    // exact positions of an object moving at 2 per frame: the velocity estimate converges to 2
    let mut e = CvEstimate { x: [0.0, 0.0], p: [[1.0, 0.0], [0.0, 1.0]] };
    for t in 1..=30 {
        e = cv_update(cv_predict(e, 1.0, 0.01), 2.0 * t as f32, 1.0).0;
    }
    assert!(close(e.x[1], 2.0, 0.05), "velocity after 30 frames {}, expected about 2", e.x[1]);
}

#[test]
fn c4_mahalanobis_b4() {
    // B4 in 1D, as a 2D innovation along x with a round S: 2.5 px with S = 4 is 1.25σ
    let d2 = mahalanobis2([2.5, 0.0], [[4.0, 0.0], [0.0, 4.0]]);
    assert!(close(d2.sqrt(), 1.25, 1e-4), "B4(a): {} standard deviations, expected 1.25", d2.sqrt());
    let d2 = mahalanobis2([2.5, 0.0], [[0.5, 0.0], [0.0, 0.5]]);
    assert!(close(d2.sqrt(), 3.536, 1e-3), "B4(b): {} standard deviations, expected 3.54", d2.sqrt());
}

#[test]
fn c4_mahalanobis_mean_is_two_b9() {
    // B9: for innovations drawn from N(0, S), the average d² is 2. S = L Lᵀ with L = [[2, 0], [1, 1]].
    let s = [[4.0, 2.0], [2.0, 2.0]];
    let mut rng = Rng::new(3);
    let n = 20000;
    let mut sum = 0.0;
    for _ in 0..n {
        let (u, v) = (rng.normal(), rng.normal());
        sum += mahalanobis2([2.0 * u, u + v], s);
    }
    let mean = sum / n as f32;
    assert!(close(mean, 2.0, 0.05), "B9: mean d² {mean}, expected about 2");
}
