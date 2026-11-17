// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p fret sensitised_emission::

use imgcourse::Image;
use fret::sensitised_emission::*;

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

fn row(v: &[f32]) -> Image<f32> {
    Image::from_vec(v.len(), 1, v.to_vec())
}

#[test]
fn c1_slopes_b1() {
    let all = Image::new(3, 1, true);
    let d = slope_through_origin(&row(&[200., 400., 800.]), &row(&[98., 205., 398.]), &all);
    assert!(close(d, 0.50, 1e-4), "B1(a): got d = {d}, expected 0.50");
    let a = slope_through_origin(&row(&[500., 1000., 1500.]), &row(&[52., 99., 149.]), &all);
    assert!(close(a, 0.0996, 1e-4), "B1(b): got a = {a}, expected 0.0996");
    let first_two = Image::from_rows(&[[true, true, false]]);
    let d2 = slope_through_origin(&row(&[200., 400., 800.]), &row(&[98., 205., 398.]), &first_two);
    assert!(close(d2, (200. * 98. + 400. * 205.) / (200. * 200. + 400. * 400.), 1e-5), "only the pixels of the mask count: got {d2}");
}

#[test]
fn c2_corrected_fret_b2_b3() {
    let (idd, ida, iaa) = (row(&[400., 1000.]), row(&[500., 0.5 * 1000. + 0.1 * 800.]), row(&[1000., 800.]));
    let fc = corrected_fret(&idd, &ida, &iaa, 0.5, 0.1);
    assert!(close(fc[(0, 0)], 200.0, 1e-3), "B2(a): got F_c = {}, expected 200", fc[(0, 0)]);
    assert!(close(fc[(1, 0)], 0.0, 1e-3), "a no-FRET pixel: got F_c = {}, expected 0", fc[(1, 0)]);
    let wrong = corrected_fret(&idd, &ida, &iaa, 0.4, 0.1);
    assert!(close(wrong[(0, 0)], 240.0, 1e-3), "B3, d = 0.4: got F_c = {}, expected 240", wrong[(0, 0)]);
    assert!(close(wrong[(1, 0)], 100.0, 1e-3), "B3, no-FRET pixel with I_DD = 1000: got false F_c = {}, expected 100", wrong[(1, 0)]);
}

#[test]
fn c2_noise_b4() {
    let s = fc_noise(400.0, 500.0, 1000.0, 0.5, 0.1);
    assert!(close(s, 24.7, 0.05), "B4(a): got {s}, expected 24.7");
    let s = fc_noise(400.0, 320.0, 1000.0, 0.5, 0.1);
    assert!(close(s, 20.7, 0.05), "B4(b): got {s}, expected 20.7");
}

#[test]
fn c3_e_app_b2_b3() {
    let e = e_app(&row(&[200., 240., -50.]), &row(&[400., 400., 10.]), 2.0);
    assert!(close(e[(0, 0)], 0.20, 1e-5), "B2(b): got E_app = {}, expected 0.20", e[(0, 0)]);
    assert!(close(e[(1, 0)], 240.0 / 1040.0, 1e-5), "B3: got E_app = {}, expected 0.23", e[(1, 0)]);
    assert_eq!(e[(2, 0)], 0.0, "a pixel with F_c + G I_DD ≤ 0 should give 0");
}

#[test]
fn c4_per_cell_b2_b5() {
    // two cells of two pixels each, every pixel as in B2; cell 2 has twice the acceptor (B5)
    let labels = Image::from_rows(&[[1u32, 1, 0, 2, 2]]);
    let fc = row(&[200., 200., 0., 200., 200.]);
    let idd = row(&[400., 400., 0., 400., 400.]);
    let iaa = row(&[1000., 1000., 0., 2000., 2000.]);
    let v = per_cell(&fc, &idd, &iaa, &labels, 2.0);
    assert_eq!(v.len(), 3, "one entry per label 0, 1, 2");
    assert_eq!(v[1].area, 2, "cell 1 has 2 pixels");
    assert!(close(v[1].e_app, 0.20, 1e-5), "B2(b), E_app: got {}", v[1].e_app);
    assert!(close(v[1].per_acceptor, 0.20, 1e-5), "B2(b), F_c / I_AA: got {}", v[1].per_acceptor);
    assert!(close(v[1].symmetric, 0.316, 1e-3), "B2(b), F_c / sqrt(I_DD I_AA): got {}", v[1].symmetric);
    assert!(close(v[2].e_app, v[1].e_app, 1e-6), "B5: E_app should not change with the acceptor level");
    assert!(close(v[2].per_acceptor, 0.10, 1e-5), "B5: F_c / I_AA should halve, got {}", v[2].per_acceptor);
    assert!(close(v[2].symmetric, 0.316 / 2f32.sqrt(), 1e-3), "B5: the symmetric index should drop by √2, got {}", v[2].symmetric);
    // ratio of sums, not mean of ratios
    let v = per_cell(&row(&[100., 0.]), &row(&[100., 900.]), &row(&[1., 1.]), &Image::from_rows(&[[1u32, 1]]), 1.0);
    assert!(close(v[1].e_app, 100.0 / 1100.0, 1e-6), "sums first: 100 / (100 + 1000) = 0.091 (the mean of the pixel values would be 0.25), got {}", v[1].e_app);
}
