// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (B1, B3 to
// B5, B7, B8) and a frame pair with a known displacement.
// Run: cargo test -p opticalflow nonrigid_reg::

use imgcourse::Image;
use opticalflow::nonrigid_reg::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_bspline3_b8() {
    for (t, e) in [(0.0, 2.0 / 3.0), (1.0, 1.0 / 6.0), (-1.0, 1.0 / 6.0), (2.0, 0.0), (0.5, 23.0 / 48.0), (-0.5, 23.0 / 48.0), (2.5, 0.0)] {
        assert!(close(bspline3(t), e, 1e-6), "β({t}): got {}, expected {e}", bspline3(t));
    }
    // Partition of unity: the four B-splines around any t add up to 1.
    for t in [0.0, 0.3, 0.77] {
        let s: f32 = (-2..=2).map(|k| bspline3(t - k as f32)).sum();
        assert!(close(s, 1.0, 1e-6), "Σ β(t − k) at t = {t}: got {s}, expected 1");
    }
}

#[test]
fn c2_ffd_displacement_b1() {
    // B1 in 2D: control points every 10 px; the column of control points at x = 20 (k = 3)
    // all moved by 6 along x. Along y they add up to 1, so d depends on x only.
    let cx = Image::from_fn(7, 7, |k, _| if k == 3 { 6.0 } else { 0.0 });
    let cy = Image::new(7, 7, 0.0);
    for (x, e) in [(20.0, 4.0), (10.0, 1.0), (30.0, 1.0), (15.0, 2.875), (0.0, 0.0), (40.0, 0.0)] {
        let (dx, dy) = ffd_displacement(&cx, &cy, 10.0, x, 22.0);
        assert!(close(dx, e, 1e-4) && dy == 0.0, "B1: d({x}) = {dx}, expected {e}");
    }
    // B1(c): all coefficients 6: a translation by 6.
    let all = Image::new(7, 7, 6.0);
    let (dx, _) = ffd_displacement(&all, &cy, 10.0, 17.3, 25.0);
    assert!(close(dx, 6.0, 1e-4), "B1(c): all coefficients 6 give d = 6 everywhere, got {dx}");
}

#[test]
fn c2_ffd_displacement_b7() {
    // B7: control points every 16 px; control point (2, 2) sits at pixel (16, 16), moved by (9, 0).
    let mut cx = Image::new(8, 8, 0.0);
    cx[(2, 2)] = 9.0;
    let cy = Image::new(8, 8, 0.0);
    for ((x, y), e) in [((16.0, 16.0), 4.0), ((32.0, 16.0), 1.0), ((32.0, 32.0), 0.25), ((48.0, 16.0), 0.0)] {
        let (dx, _) = ffd_displacement(&cx, &cy, 16.0, x, y);
        assert!(close(dx, e, 1e-4), "B7: pixel ({x}, {y}) moves by {dx}, expected {e}");
    }
}

/// A displacement field given as a function of (x, y).
type Field = fn(f32, f32) -> (f32, f32);

#[test]
fn c3_jacobian_b3() {
    let field = |f: Field| {
        (Image::from_fn(9, 9, |x, y| f(x as f32, y as f32).0), Image::from_fn(9, 9, |x, y| f(x as f32, y as f32).1))
    };
    let cases: [(&str, Field, f32); 4] = [
        ("(a) d = (0.1x, 0)", |x, _| (0.1 * x, 0.0), 1.1),
        ("(b) d = (0.1x, 0.1y)", |x, y| (0.1 * x, 0.1 * y), 1.21),
        ("(c) d = (0.5y, 0)", |_, y| (0.5 * y, 0.0), 1.0),
        ("(d) d = (−1.2x, 0)", |x, _| (-1.2 * x, 0.0), -0.2),
    ];
    for (name, f, e) in cases {
        let (dx, dy) = field(f);
        let j = jacobian_determinant(&dx, &dy)[(4, 4)];
        assert!(close(j, e, 1e-4), "B3 {name}: got {j}, expected {e}");
    }
}

#[test]
fn c4_entropy_b5() {
    for (p, e) in [(vec![1.0], 0.0), (vec![0.5, 0.5], 1.0), (vec![0.25; 4], 2.0), (vec![0.5, 0.25, 0.25, 0.0], 1.5)] {
        assert!(close(entropy(&p), e, 1e-5), "B5: H({p:?}) = {}, expected {e}", entropy(&p));
    }
}

#[test]
fn c4_mutual_information_b4() {
    let row = |v: [f32; 8]| Image::from_rows(&[v]);
    let a = row([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let cases = [
        ("(a) B = 1 − A", row([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 1.0),
        ("(b) all joint counts 2", row([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]), 0.0),
        ("(c) one pixel wrong", row([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.549),
    ];
    for (name, b, e) in cases {
        let mi = mutual_information(&a, &b, 2);
        assert!(close(mi, e, 1e-3), "B4 {name}: MI = {mi}, expected {e}");
    }
}

/// A smooth pattern moved by (dx, dy).
fn pattern(size: usize, dx: f32, dy: f32) -> Image<f32> {
    Image::from_fn(size, size, |x, y| {
        let (x, y) = (x as f32 - dx, y as f32 - dy);
        0.5 + 0.2 * (x / 5.0).sin() + 0.2 * (y / 6.0 + 0.3 * x / 7.0).cos()
    })
}

#[test]
fn c5_register_ssd_finds_a_shift() {
    // The moving image is the fixed one moved by (1.5, −1): M(x + d) = F(x) for d = (1.5, −1).
    let (fixed, moving) = (pattern(80, 0.0, 0.0), pattern(80, 1.5, -1.0));
    let (cx, cy) = register_ssd(&fixed, &moving, 16.0, 200, 0.05);
    let (dx, dy) = ffd_displacement(&cx, &cy, 16.0, 40.0, 40.0);
    assert!(close(dx, 1.5, 0.1) && close(dy, -1.0, 0.1), "registration of a shift by (1.5, −1): got d = ({dx:.2}, {dy:.2}) at the centre");
}
