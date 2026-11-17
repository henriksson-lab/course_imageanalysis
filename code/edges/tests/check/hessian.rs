// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p edges hessian::

use edges::hessian::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_second_differences_b2() {
    // B2: rows y = −1, 0, 1 from top to bottom; the centre is (1, 1).
    let patch = Image::from_rows(&[[9., 5., 1.], [5., 9., 5.], [1., 5., 9.]]);
    let h = hessian_at(&patch, 1, 1);
    assert!(close(h[0], -8.0, 1e-5) && close(h[1], 4.0, 1e-5) && close(h[2], -8.0, 1e-5), "B2(a): got [Lxx, Lxy, Lyy] = {h:?}, expected [−8, 4, −8]");
}

#[test]
fn c1_hessian_of_quadratics_b1() {
    // B1 on pixels: f = −x² and f = −(x + y)², around the centre of a 21 × 21 image. The
    // Hessian of a quadratic is the same at every scale, times σ² for the normalisation.
    let sigma = 2.0;
    let ridge = Image::from_fn(21, 21, |x, _| -((x as f32 - 10.0).powi(2)));
    let h = hessian(&ridge, sigma)[(10, 10)].map(|v| v / (sigma * sigma));
    assert!(close(h[0], -2.0, 1e-3) && close(h[1], 0.0, 1e-3) && close(h[2], 0.0, 1e-3), "B1(a), f = −x²: got H/σ² = {h:?}, expected [−2, 0, 0]");
    let diagonal = Image::from_fn(21, 21, |x, y| -((x as f32 - 10.0) + (y as f32 - 10.0)).powi(2));
    let h = hessian(&diagonal, sigma)[(10, 10)].map(|v| v / (sigma * sigma));
    assert!(close(h[0], -2.0, 1e-3) && close(h[1], -2.0, 1e-3) && close(h[2], -2.0, 1e-3), "B1(d), f = −(x + y)²: got H/σ² = {h:?}, expected [−2, −2, −2]");
}

#[test]
fn c2_eigen_b1_b2_b5() {
    // (h, λ1, λ2, eigenvector of λ1 up to sign)
    let s = 0.5f32.sqrt();
    let cases = [
        ([-2.0, 0.0, 0.0], 0.0, -2.0, [0.0, 1.0]), // B1(a): a ridge along y
        ([2.0, 0.0, -2.0], 2.0, -2.0, [1.0, 0.0]), // B1(c): a saddle (equal sizes: either order)
        ([-2.0, -2.0, -2.0], 0.0, -4.0, [s, -s]),  // B1(d): a ridge along (1, −1)
        ([-8.0, 4.0, -8.0], -4.0, -12.0, [s, s]),  // B2(b)
    ];
    for (h, e1, e2, u) in cases {
        let (l1, l2, v) = eigen_sym(h);
        let both = (close(l1, e1, 1e-4) && close(l2, e2, 1e-4)) || (e1.abs() == e2.abs() && close(l1, e2, 1e-4) && close(l2, e1, 1e-4));
        assert!(both, "H = {h:?}: got λ1 = {l1}, λ2 = {l2}, expected {e1} and {e2}");
        assert!(l1.abs() <= l2.abs(), "H = {h:?}: λ1 = {l1} should be the smaller in size than λ2 = {l2}");
        if e1.abs() != e2.abs() {
            let dot = (v[0] * u[0] + v[1] * u[1]).abs();
            assert!(close(dot, 1.0, 1e-4), "H = {h:?}: eigenvector of λ1 {v:?}, expected ±{u:?}");
        }
    }
    // B5
    let (l1, l2, _) = eigen_sym([-20.0, 2.0, -3.0]);
    assert!(close(l1, -2.77, 0.01) && close(l2, -20.23, 0.01), "B5: got λ1 = {l1}, λ2 = {l2}, expected −2.77 and −20.23");
    // A multiple of the identity: every direction is an eigenvector, any unit vector will do.
    let (l1, l2, v) = eigen_sym([-2.0, 0.0, -2.0]);
    assert!(close(l1, -2.0, 1e-6) && close(l2, -2.0, 1e-6) && close(v[0].hypot(v[1]), 1.0, 1e-5), "B1(b): got λ = {l1}, {l2}, u = {v:?}");
}

#[test]
fn c3_classify_b1_b2() {
    let cases = [
        (0.0, -2.0, Class::BrightRidge), // B1(a)
        (-2.0, -2.0, Class::BrightBlob), // B1(b)
        (2.0, -2.0, Class::Saddle),      // B1(c)
        (-4.0, -12.0, Class::BrightRidge), // B2(c): ratio 1/3
        (5.0, 9.0, Class::DarkBlob),
        (0.5, 9.0, Class::DarkRidge),
        (0.1, 0.5, Class::Flat),
    ];
    for (l1, l2, class) in cases {
        let got = classify(l1, l2, 1.0);
        assert_eq!(got, class, "λ = ({l1}, {l2}), flat below 1: got {got:?}, expected {class:?}");
    }
}

#[test]
fn c3_disc_at_two_scales_b4() {
    // B4: a bright disc of radius 10 (100 on 0). Classes at 0, 8, 12 and 20 pixels from the centre.
    let disc = Image::from_fn(81, 41, |x, y| if (x as f32 - 40.0).powi(2) + (y as f32 - 20.0).powi(2) <= 100.0 { 100.0 } else { 0.0 });
    let expected = [
        (2.0, [Class::Flat, Class::BrightRidge, Class::DarkRidge, Class::Flat]),
        (7.0, [Class::BrightBlob, Class::BrightRidge, Class::Saddle, Class::DarkRidge]),
    ];
    for (sigma, classes) in expected {
        let h = hessian(&disc, sigma);
        for (d, class) in [0, 8, 12, 20].iter().zip(classes) {
            let (l1, l2, _) = eigen_sym(h[(40 + d, 20)]);
            let got = classify(l1, l2, 4.0);
            assert_eq!(got, class, "B4, σ = {sigma}, {d} pixels from the centre: λ = ({l1:.1}, {l2:.1}) gives {got:?}, expected {class:?}");
        }
    }
}

#[test]
fn c4_vesselness_b3_b5() {
    let cases = [((-10.0, -10.0), 0.05), ((0.0, -20.0), 0.59), ((-2.0, -20.0), 0.58), ((-2.77, -20.23), 0.58), ((1.0, 20.0), 0.0)];
    for ((l1, l2), v) in cases {
        let got = vesselness(l1, l2, 0.5, 15.0);
        assert!(close(got, v, 0.006), "B3/B5, λ = ({l1}, {l2}), β = 0.5, c = 15: got {got:.3}, expected {v}");
    }
    let strict = vesselness(-2.77, -20.23, 0.1, 15.0);
    let weak = vesselness(-2.77, -20.23, 0.5, 50.0);
    assert!(close(strict, 0.24, 0.01) && close(weak, 0.08, 0.005), "B5: β = 0.1 gives {strict:.3} (expected 0.24), c = 50 gives {weak:.3} (expected 0.08)");
}

#[test]
fn c5_best_scale_follows_width() {
    // B8: a line with a Gaussian profile of width s has its best scale near √2·s.
    let sigmas: Vec<f32> = (0..15).map(|k| 1.0 + 0.5 * k as f32).collect();
    for s in [1.5f32, 3.0] {
        let line = Image::from_fn(61, 9, |x, _| 100.0 * (-((x as f32 - 30.0).powi(2)) / (2.0 * s * s)).exp());
        let (v, best) = multiscale_vesselness(&line, &sigmas, 0.5, 15.0, true);
        let b = best[(30, 4)];
        assert!(v[(30, 4)] > 0.5, "line of width {s}: vesselness at its centre only {}", v[(30, 4)]);
        assert!(close(b, 2f32.sqrt() * s, 0.6), "line of width {s}: best σ = {b}, expected about √2·{s} = {:.2}", 2f32.sqrt() * s);
        let (dark, _) = multiscale_vesselness(&line, &sigmas, 0.5, 15.0, false);
        assert!(dark[(30, 4)] == 0.0, "a bright line should give 0 when looking for dark lines, got {}", dark[(30, 4)]);
    }
}
