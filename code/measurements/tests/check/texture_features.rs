// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p measurements texture_features::

use imgcourse::Image;
use measurements::texture_features::*;

/// The 4 × 4 image of the exercise, levels 0, 1, 2 (B1).
fn b1_image() -> Image<u32> {
    Image::from_rows(&[[0, 0, 1, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 2, 2]])
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn same_matrix(got: &[Vec<f32>], expected: &[&[f32]], what: &str) {
    let ok = got.len() == expected.len() && got.iter().zip(expected).all(|(g, e)| g.len() == e.len() && g.iter().zip(*e).all(|(a, b)| close(*a, *b, 1e-6)));
    assert!(ok, "{what}: got {got:?}, expected {expected:?}");
}

#[test]
fn c1_quantise() {
    let img = Image::from_rows(&[[0.0, 0.124, 0.125, 0.5, 0.99, 1.0, -0.2, 1.3]]);
    let q = quantise(&img, 8, 0.0, 1.0);
    assert_eq!(q.row(0), &[0, 0, 1, 4, 7, 7, 0, 7], "8 levels on 0..1 (the last level includes 1; values outside are clipped)");
    let q = quantise(&Image::from_rows(&[[10.0, 20.0, 30.0]]), 2, 10.0, 30.0);
    assert_eq!(q.row(0), &[0, 1, 1], "2 levels on 10..30");
}

#[test]
fn c2_glcm_0_degrees_b1() {
    let p = glcm(&b1_image(), 3, 1, 0, None);
    same_matrix(&p, &[&[4., 2., 1.], &[2., 4., 0.], &[1., 0., 10.]], "B1(a) 0°, d = 1");
}

#[test]
fn c2_glcm_90_degrees_b2() {
    let p = glcm(&b1_image(), 3, 0, -1, None);
    same_matrix(&p, &[&[6., 0., 2.], &[0., 4., 2.], &[2., 2., 6.]], "B2(a) 90° (partner above), d = 1");
}

#[test]
fn c2_glcm_45_degrees_b3() {
    let p = glcm(&b1_image(), 3, 1, -1, None);
    same_matrix(&p, &[&[4., 1., 0.], &[1., 2., 2.], &[0., 2., 6.]], "B3 45° (partner up and to the right), d = 1: only 9 pairs");
}

#[test]
fn c2_glcm_mask() {
    // Only pairs with both pixels in the mask count: the left two columns of B1.
    let mask = Image::from_fn(4, 4, |x, _| x < 2);
    let p = glcm(&b1_image(), 3, 1, 0, Some(&mask));
    same_matrix(&p, &[&[4., 0., 1.], &[0., 0., 0.], &[1., 0., 2.]], "0° inside a mask of the two left columns");
}

#[test]
fn c3_haralick_b1() {
    let h = haralick(&glcm(&b1_image(), 3, 1, 0, None));
    assert!(close(h.contrast, 0.5, 1e-4), "B1(b) contrast: got {}, expected 0.5", h.contrast);
    assert!(close(h.homogeneity, 0.85, 1e-4), "B1(b) homogeneity: got {}, expected 0.85", h.homogeneity);
    assert!(close(h.asm, 142.0 / 576.0, 1e-4), "B1(b) ASM: got {}, expected 0.247", h.asm);
    assert!(close(h.entropy, 2.37, 0.005), "B1(c) entropy: got {}, expected 2.37 bits", h.entropy);
    assert!(close(h.correlation, 0.654, 0.002), "B1(d) correlation: got {}, expected 0.654", h.correlation);
    let h = haralick(&glcm(&b1_image(), 3, 0, -1, None));
    assert!(close(h.contrast, 20.0 / 24.0, 1e-4) && close(h.homogeneity, 18.8 / 24.0, 1e-4), "B2(b) 90°: got contrast {}, homogeneity {}, expected 0.83, 0.78", h.contrast, h.homogeneity);
}

#[test]
fn c3_same_histogram_b4() {
    let checker = Image::from_fn(4, 4, |x, y| ((x + y) % 2) as u32);
    let halves = Image::from_fn(4, 4, |x, _| (x >= 2) as u32);
    let c = haralick(&glcm(&checker, 2, 1, 0, None));
    let h = haralick(&glcm(&halves, 2, 1, 0, None));
    assert!(close(c.contrast, 1.0, 1e-5) && close(c.correlation, -1.0, 1e-5), "B4(b) checkerboard: got contrast {}, correlation {}, expected 1, −1", c.contrast, c.correlation);
    assert!(close(h.contrast, 1.0 / 3.0, 1e-5) && close(h.correlation, 1.0 / 3.0, 1e-5), "B4(b) halves: got contrast {}, correlation {}, expected 0.33, 0.33", h.contrast, h.correlation);
    assert!(close(c.entropy, 1.0, 1e-4) && close(h.entropy, 1.918, 1e-3), "B4(b) entropy: got {} and {}, expected 1.0 and 1.92", c.entropy, h.entropy);
}

#[test]
fn c3_haralick_45_degrees_b3() {
    let h = haralick(&glcm(&b1_image(), 3, 1, -1, None));
    assert!(close(h.contrast, 6.0 / 18.0, 1e-4), "B3 contrast at 45°: got {}, expected 0.33", h.contrast);
}

#[test]
fn c4_lbp_b6() {
    let n = Image::from_rows(&[[6.0, 2.0, 9.0], [5.0, 5.0, 8.0], [1.0, 4.0, 7.0]]);
    let code = lbp(&n, 1, 1);
    assert_eq!(code, 185, "B6(a): got {code} = {code:08b}, expected 185 = 10111001");
    assert_eq!(rotation_invariant(code), 55, "B6(b) rotation-invariant code");
    let brighter = n.map(|v| 2.0 * v + 10.0);
    assert_eq!(lbp(&brighter, 1, 1), 185, "B6(d) an offset and a gain do not change the code");
    let flat = Image::new(3, 3, 0.5);
    assert_eq!(lbp(&flat, 1, 1), 255, "a flat patch: every neighbour is at least the centre");
}
