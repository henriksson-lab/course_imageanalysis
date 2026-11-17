// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (B1 to B4, B9).
// Run: cargo test -p opticalflow flow_constraint::

use opticalflow::flow_constraint::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const F1: [f32; 7] = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0];
const F2: [f32; 7] = [0.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0];

#[test]
fn c1_derivatives_b1a() {
    let (ix, it) = derivatives_1d(&F1, &F2);
    let (eix, eit) = ([0.0, 0.5, 2.0, 0.0, -2.0, -0.5, 0.0], [0.0, 0.0, -1.0, -3.0, 3.0, 1.0, 0.0]);
    for x in 0..7 {
        assert!(close(ix[x], eix[x], 1e-6), "B1(a) Ix at x = {x}: got {}, expected {}", ix[x], eix[x]);
        assert!(close(it[x], eit[x], 1e-6), "B1(a) It at x = {x}: got {}, expected {}", it[x], eit[x]);
    }
}

#[test]
fn c2_pixel_speed_b1b() {
    let (ix, it) = derivatives_1d(&F1, &F2);
    let expected = [Some(0.0), Some(0.5), None, Some(1.5), Some(2.0)];
    for x in 1..=5 {
        let got = pixel_speed(ix[x], it[x]);
        let ok = match (got, expected[x - 1]) {
            (Some(g), Some(e)) => close(g, e, 1e-6),
            (None, None) => true,
            _ => false,
        };
        assert!(ok, "B1(b) at x = {x}: got {got:?}, expected {:?}", expected[x - 1]);
    }
}

#[test]
fn c3_window_speed_b1c_b2() {
    let (ix, it) = derivatives_1d(&F1, &F2);
    let u = window_speed(&ix, &it, 3, 2).unwrap();
    assert!(close(u, 1.0, 1e-5), "B1(c): got {u}, expected 1");
    // B2(a): moved by 2 pixels.
    let (ix, it) = derivatives_1d(&F1, &[0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0]);
    let u = window_speed(&ix, &it, 3, 2).unwrap();
    assert!(close(u, 4.0 / 8.5, 1e-4), "B2(a): got {u}, expected 0.47");
    // B2(b): moved by 1 pixel, 10 % dimmer.
    let (ix, it) = derivatives_1d(&F1, &F2.map(|v| 0.9 * v));
    let u = window_speed(&ix, &it, 3, 2).unwrap();
    assert!(close(u, 7.65 / 8.5, 1e-4), "B2(b): got {u}, expected 0.90");
    // A window that reaches past the ends of the profile, and a flat one.
    let (ix, it) = derivatives_1d(&F1, &F2);
    assert!(window_speed(&ix, &it, 0, 10).is_some(), "the window may reach past the ends of the profile");
    assert!(window_speed(&[0.0; 5], &[1.0; 5], 2, 1).is_none(), "a flat window has no speed");
}

#[test]
fn c4_normal_flow_b9_b3() {
    let (u, v) = normal_flow(3.0, 4.0, -10.0).unwrap();
    assert!(close(u, 1.2, 1e-5) && close(v, 1.6, 1e-5), "B9: got ({u}, {v}), expected (1.2, 1.6)");
    let (u, v) = normal_flow(1.0, 0.0, -2.0).unwrap();
    assert!(close(u, 2.0, 1e-6) && close(v, 0.0, 1e-6), "B3(b) pixel A: got ({u}, {v}), expected (2, 0)");
    assert!(normal_flow(0.0, 0.0, 1.0).is_none(), "no gradient: no normal flow");
}

#[test]
fn c5_intersect_b3_b4() {
    let (u, v) = intersect([1.0, 0.0, -2.0], [0.0, 1.0, -1.0]).unwrap();
    assert!(close(u, 2.0, 1e-5) && close(v, 1.0, 1e-5), "B3: got ({u}, {v}), expected (2, 1)");
    let (u, v) = intersect([2.0, 1.0, -5.0], [1.0, -2.0, 0.0]).unwrap();
    assert!(close(u, 2.0, 1e-5) && close(v, 1.0, 1e-5), "B4: got ({u}, {v}), expected (2, 1)");
    assert!(intersect([1.0, 0.0, -2.0], [2.0, 0.0, -4.0]).is_none(), "B3(c): A and C are the same line, no crossing");
}
