// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (A2 to A4).
// Run: cargo test -p keypoints icp::

use keypoints::icp::*;
use imgcourse::Rng;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The pairs of A2: p and its image under a turn by 36.87° (cos 0.8, sin 0.6) and a shift (10, 5).
const P: [Point; 4] = [(2.0, 0.0), (-2.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
const Q: [Point; 4] = [(11.6, 6.2), (8.4, 3.8), (9.4, 5.8), (10.6, 4.2)];

fn on_x(xs: &[f32]) -> Vec<Point> {
    xs.iter().map(|&x| (x, 0.0)).collect()
}

#[test]
fn c1_centroid_a2() {
    let (x, y) = centroid(&Q);
    assert!(close(x, 10.0, 1e-5) && close(y, 5.0, 1e-5), "A2: q̄ = ({x}, {y}), expected (10, 5)");
    let (x, y) = centroid(&P);
    assert!(close(x, 0.0, 1e-6) && close(y, 0.0, 1e-6), "A2: p̄ = ({x}, {y}), expected (0, 0)");
}

#[test]
fn c2_fit_rigid_a2() {
    let t = fit_rigid(&P, &Q);
    assert!(close(t.angle.to_degrees(), 36.8699, 1e-3), "A2: angle {}°, expected atan2(6, 8) = 36.87°", t.angle.to_degrees());
    assert!(close(t.tx, 10.0, 1e-4) && close(t.ty, 5.0, 1e-4), "A2: shift ({}, {}), expected (10, 5)", t.tx, t.ty);
    // Random sets: the fit recovers a known transform exactly, and the shift moves p̄ onto q̄.
    let mut rng = Rng::new(3);
    let truth = Rigid { angle: -1.1, tx: 30.0, ty: -12.0 };
    let p: Vec<Point> = (0..12).map(|_| (100.0 * rng.uniform(), 100.0 * rng.uniform())).collect();
    let q: Vec<Point> = p.iter().map(|&x| truth.apply(x)).collect();
    let t = fit_rigid(&p, &q);
    assert!(close(t.angle, truth.angle, 1e-4) && close(t.tx, truth.tx, 1e-2) && close(t.ty, truth.ty, 1e-2), "exact pairs: got {t:?}, expected {truth:?}");
}

#[test]
fn c3_nearest() {
    let set = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
    let (j, d) = nearest((7.0, 1.0), &set);
    assert!(j == 1 && close(d, 10.0f32.sqrt(), 1e-5), "(7, 1): nearest {j} at {d}, expected 1 at √10");
    let (j, _) = nearest((4.0, 0.0), &on_x(&[3.0, 7.0, 11.0]));
    assert_eq!(j, 0, "A3: 4 is nearest to 3");
}

#[test]
fn c4_icp_step_a3() {
    // A3, first iteration from t = 0: 0 → 3, 4 → 3, 8 → 7; the fit moves by 1/3.
    let s = icp_step(&on_x(&[0.0, 4.0, 8.0]), &on_x(&[3.0, 7.0, 11.0]), &Rigid::IDENTITY, 1.0);
    assert_eq!(s.matches, vec![Some(0), Some(0), Some(1)], "A3: the first matching");
    assert!(close(s.rms, (11.0f32 / 3.0).sqrt(), 1e-4), "A3: RMS before the fit √(11/3), got {}", s.rms);
    assert!(close(s.transform.tx, 1.0 / 3.0, 1e-4) && close(s.transform.angle, 0.0, 1e-5), "A3: t = 1/3, got {:?}", s.transform);
    // A4, trimmed (keep 3 of 4): the stray point 30 is left out.
    let s = icp_step(&on_x(&[0.0, 2.0, 4.0, 30.0]), &on_x(&[0.5, 2.5, 4.5]), &Rigid::IDENTITY, 0.75);
    assert_eq!(s.matches, vec![Some(0), Some(1), Some(2), None], "A4: trimming drops the pair of 30");
    assert!(close(s.transform.tx, 0.5, 1e-4), "A4 trimmed: t = 0.5, got {:?}", s.transform);
}

#[test]
fn c5_icp_a3_a4() {
    let (p, q) = (on_x(&[0.0, 4.0, 8.0]), on_x(&[3.0, 7.0, 11.0]));
    let (t, rms) = icp(&p, &q, Rigid::IDENTITY, 50, 1.0);
    assert!(close(t.tx, 1.0 / 3.0, 1e-4), "A3 from 0: stuck at t = 1/3, got {t:?}");
    assert_eq!(rms.len(), 2, "A3 from 0: the second matching is the same as the first, so ICP stops after 2 iterations");
    let (t, _) = icp(&p, &q, Rigid { angle: 0.0, tx: 2.0, ty: 0.0 }, 50, 1.0);
    assert!(close(t.tx, 3.0, 1e-4), "A3 from 2: t = 3, got {t:?}");
    let (p, q) = (on_x(&[0.0, 2.0, 4.0, 30.0]), on_x(&[0.5, 2.5, 4.5]));
    let (t, _) = icp(&p, &q, Rigid::IDENTITY, 50, 1.0);
    assert!(close(t.tx, -7.5, 1e-4), "A4 all pairs: t = −6, then −7.5; got {t:?}");
    let (t, _) = icp(&p, &q, Rigid::IDENTITY, 50, 0.75);
    assert!(close(t.tx, 0.5, 1e-4), "A4 trimmed: t = 0.5, got {t:?}");
    // A small turn of a random set is found again from no rotation.
    let mut rng = Rng::new(9);
    let fixed: Vec<Point> = (0..40).map(|_| (200.0 * rng.uniform(), 200.0 * rng.uniform())).collect();
    let back = Rigid { angle: -0.05, tx: 3.0, ty: -2.0 };
    let moving: Vec<Point> = fixed.iter().map(|&p| back.apply(p)).collect();
    let (t, rms) = icp(&moving, &fixed, Rigid::IDENTITY, 50, 1.0);
    assert!(close(t.angle, 0.05, 1e-3) && rms.last().unwrap() < &1e-2, "a 3° turn: got {t:?}, RMS {rms:?}");
}
