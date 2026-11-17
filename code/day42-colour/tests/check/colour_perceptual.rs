// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day42 colour_perceptual::

use day42::colour_perceptual::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn hsv_codes(c: [f32; 3]) -> [f32; 3] {
    rgb_to_hsv(c.map(|v| v / 255.0))
}

#[test]
fn c1_hsv_b1() {
    for (c, e) in [([255., 0., 0.], [0., 1., 1.]), ([0., 128., 0.], [120., 1., 0.502]), ([200., 100., 50.], [20., 0.75, 0.784])] {
        let got = hsv_codes(c);
        assert!((0..3).all(|i| close(got[i], e[i], 2e-3)), "{c:?}: got (H, S, V) = {got:?}, expected {e:?}");
    }
    let grey = hsv_codes([128., 128., 128.]);
    assert!(close(grey[1], 0.0, 1e-6), "a grey: got saturation {}, expected 0", grey[1]);
    let black = rgb_to_hsv([0.0; 3]);
    assert!(black.iter().all(|v| v.is_finite()), "black: got {black:?}, expected no NaN");
}

#[test]
fn c1_shading_b3() {
    let light = rgb_to_hsv([0.60, 0.10, 0.08]);
    let shadow = rgb_to_hsv([0.30, 0.05, 0.04]);
    assert!(close(light[0], 2.3, 0.05) && close(light[1], 0.867, 1e-3), "light (B3a): got {light:?}, expected (2.3°, 0.867, 0.60)");
    assert!(close(shadow[0], light[0], 1e-3) && close(shadow[1], light[1], 1e-4), "shadow: got {shadow:?}, expected the same H and S as {light:?}");
    assert!(close(shadow[2], 0.30, 1e-6), "shadow: got V = {}, expected 0.30", shadow[2]);
}

#[test]
fn c1_noisy_grey_b4() {
    let a = rgb_to_hsv([0.50, 0.52, 0.49]);
    let b = rgb_to_hsv([0.50, 0.49, 0.51]);
    assert!(close(a[0], 100.0, 0.1) && close(a[1], 0.058, 1e-3), "(0.50, 0.52, 0.49): got {a:?}, expected H 100°, S 0.058");
    assert!(close(b[0], 270.0, 0.1) && close(b[1], 0.039, 1e-3), "(0.50, 0.49, 0.51): got {b:?}, expected H 270°, S 0.039");
    let c = rgb_to_hsv([1.0, 0.0, 0.2]);
    assert!(close(c[0], 348.0, 0.1), "magenta-red (1, 0, 0.2): got H {}, expected 348° (not negative)", c[0]);
}

#[test]
fn c2_circular_mean_b2() {
    let (m, r) = circular_mean(&[350.0, 10.0]);
    assert!(close(m, 0.0, 1e-3) || close(m, 360.0, 1e-3), "mean of 350° and 10° (B2a): got {m}, expected 0°");
    assert!(close(r, 0.985, 1e-3), "its length: got {r}, expected 0.985");
    let (_, r) = circular_mean(&[90.0, 270.0]);
    assert!(r < 1e-5, "90° and 270° (B2b): got length {r}, expected 0");
    let (m, _) = circular_mean(&[300.0, 340.0]);
    assert!(close(m, 320.0, 1e-3), "300° and 340°: got {m}, expected 320°");
}

#[test]
fn c3_lab_b5() {
    let white = srgb_to_lab([1.0, 1.0, 1.0]);
    assert!(close(white[0], 100.0, 0.05) && white[1].abs() < 0.05 && white[2].abs() < 0.05, "white: got {white:?}, expected (100, 0, 0)");
    // a grey with 18 % of white's luminance (B5a): stored value encode(0.18) = 0.4613
    let grey = srgb_to_lab([0.4613; 3]);
    assert!(close(grey[0], 49.5, 0.1), "18 % grey: got L* = {}, expected 49.5", grey[0]);
    let yellow = srgb_to_lab([1.0, 1.0, 0.0]);
    let e = [97.1, -21.6, 94.5];
    assert!((0..3).all(|i| close(yellow[i], e[i], 0.2)), "yellow (255, 255, 0): got {yellow:?}, expected {e:?}");
    let green = srgb_to_lab([0.0, 80.0 / 255.0, 0.0]);
    let e = [28.7, -37.2, 35.9];
    assert!((0..3).all(|i| close(green[i], e[i], 0.2)), "dark green (0, 80, 0): got {green:?}, expected {e:?}");
}

#[test]
fn c3_delta_e_b5() {
    let d = delta_e([50.0, 20.0, 10.0], [55.0, 17.0, 14.0]);
    assert!(close(d, 7.07, 0.01), "B5(b): got {d}, expected 7.07");
    let d = delta_e(srgb_to_lab([0.0, 80.0 / 255.0, 0.0]), srgb_to_lab([0.0, 120.0 / 255.0, 0.0]));
    assert!(close(d, 22.4, 0.2), "dark greens (B5c): got {d}, expected 22.4");
}

#[test]
fn c4_hue_mask_wraps_b2c() {
    // hues 0°, 10°, 30°, 180°, 350° (full saturation), and a dark red
    let img = Image::from_rows(&[[[1.0f32, 0.0, 0.0], [1.0, 1. / 6., 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1. / 6.], [0.05, 0.0, 0.0]]]);
    let m = hue_mask(&img, 340.0, 20.0, 0.4, 0.1);
    let expected = [true, true, false, false, true, false];
    for x in 0..6 {
        assert!(m[(x, 0)] == expected[x], "pixel {x} {:?}: got {}, expected {}", img[(x, 0)], m[(x, 0)], expected[x]);
    }
    let m = hue_mask(&img, 20.0, 200.0, 0.0, 0.0);
    assert!(!m[(0, 0)] && m[(2, 0)] && m[(3, 0)], "the interval 20°–200° (no wrap): got {:?}", (0..6).map(|x| m[(x, 0)]).collect::<Vec<_>>());
}

#[test]
fn c5_kmeans_b9() {
    let points: Vec<[f32; 3]> = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0].iter().map(|&v| [v, 0.0, 0.0]).collect();
    let (c, labels) = kmeans(&points, &[[0.0; 3], [1.0, 0.0, 0.0]], 10);
    assert!(close(c[0][0], 1.0, 1e-5) && close(c[1][0], 9.0, 1e-5), "B9: got centres {} and {}, expected 1 and 9", c[0][0], c[1][0]);
    assert!(labels == vec![0, 0, 0, 1, 1, 1], "B9: got labels {labels:?}, expected [0, 0, 0, 1, 1, 1]");
    let (c, _) = kmeans(&points, &[[0.0; 3], [1.0, 0.0, 0.0]], 1);
    assert!(close(c[0][0], 0.0, 1e-5) && close(c[1][0], 6.0, 1e-5), "B9 after one step: got {} and {}, expected 0 and 6", c[0][0], c[1][0]);
}
