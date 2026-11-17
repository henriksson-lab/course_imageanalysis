// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p rgb-linear

use imgcourse::Image;
use rgb_linear::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_decode_b1a() {
    for (code, expected) in [(0.0, 0.0), (128.0, 0.216), (188.0, 0.503), (255.0, 1.0)] {
        let got = srgb_decode(code / 255.0);
        assert!(close(got, expected, 1e-3), "code {code}: got {got}, expected {expected}");
    }
}

#[test]
fn c1_encode_b1() {
    let got = 255.0 * srgb_encode(0.5);
    assert!(close(got, 187.5, 0.3), "light 0.5 (B1b): got code {got}, expected about 188");
    let got = 255.0 * srgb_encode(0.5 * srgb_decode(200.0 / 255.0));
    assert!(close(got, 146.0, 0.6), "half the light of code 200 (B1c): got code {got}, expected about 146");
    for i in 0..=20 {
        let c = i as f32 / 20.0;
        let back = srgb_encode(srgb_decode(c));
        assert!(close(back, c, 1e-5), "encode(decode({c})) = {back}, expected {c}");
    }
}

#[test]
fn c2_black_and_white_b1b() {
    let img = Image::from_rows(&[[[0.0f32; 3], [1.0; 3]], [[0.0; 3], [1.0; 3]]]);
    let codes = downsample(&img, 2, false);
    let light = downsample(&img, 2, true);
    assert!(codes.width() == 1 && codes.height() == 1, "a 2 × 2 image by 2: got {} × {}, expected 1 × 1", codes.width(), codes.height());
    assert!(close(255.0 * codes[(0, 0)][0], 127.5, 0.1), "on codes: got code {}, expected 127.5", 255.0 * codes[(0, 0)][0]);
    assert!(close(255.0 * light[(0, 0)][1], 187.5, 0.3), "in linear light: got code {}, expected about 188", 255.0 * light[(0, 0)][1]);
}

#[test]
fn c2_uniform_stays() {
    let img = Image::new(8, 6, [0.2f32, 0.5, 0.9]);
    for linear in [false, true] {
        let out = downsample(&img, 3, linear);
        assert!(out.width() == 2 && out.height() == 2, "8 × 6 by 3: got {} × {}, expected 2 × 2", out.width(), out.height());
        let p = out[(1, 1)];
        assert!(close(p[0], 0.2, 1e-4) && close(p[1], 0.5, 1e-4) && close(p[2], 0.9, 1e-4), "a uniform image must stay the same (linear = {linear}): got {p:?}");
    }
}

#[test]
fn c3_toy_display_b2() {
    let m = rgb_to_xyz_matrix([[0.6, 0.3], [0.3, 0.6], [0.2, 0.1]], [1.0 / 3.0, 1.0 / 3.0]);
    let expected = [[0.433, 0.333, 0.233], [0.217, 0.667, 0.117], [0.072, 0.111, 0.817]];
    for r in 0..3 {
        for c in 0..3 {
            assert!(close(m[r][c], expected[r][c], 2e-3), "row {}, column {}: got {}, expected {} (B2c)", r + 1, c + 1, m[r][c], expected[r][c]);
        }
    }
}

#[test]
fn c3_srgb_matrix() {
    let m = rgb_to_xyz_matrix([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], [0.3127, 0.3290]);
    let lum = [0.2126, 0.7152, 0.0722];
    for k in 0..3 {
        assert!(close(m[1][k], lum[k], 1e-3), "sRGB luminance weight {k}: got {}, expected {}", m[1][k], lum[k]);
    }
    assert!(close(m[0][0], 0.4124, 1e-3) && close(m[2][2], 0.9505, 2e-3), "sRGB matrix: got {m:?}");
}

#[test]
fn c4_ycbcr_b3() {
    let red = ycbcr([255.0, 0.0, 0.0]);
    let e = [76.2, 85.0, 255.5];
    assert!((0..3).all(|i| close(red[i], e[i], 0.2)), "pure red: got {red:?}, expected {e:?}");
    let grey = ycbcr([128.0, 128.0, 128.0]);
    assert!((0..3).all(|i| close(grey[i], 128.0, 1e-3)), "mid-grey: got {grey:?}, expected (128, 128, 128)");
}

#[test]
fn c5_grey_world_b5() {
    // An image whose channel means are (0.30, 0.20, 0.10), with the pixel of B5(b) in it.
    let img = Image::from_rows(&[[[0.45f32, 0.30, 0.08], [0.15, 0.10, 0.12]]]);
    let (g, out) = grey_world(&img);
    let e = [0.667, 1.0, 2.0];
    assert!((0..3).all(|i| close(g[i], e[i], 1e-3)), "gains (B5a): got {g:?}, expected {e:?}");
    let p = out[(0, 0)];
    let e = [0.30, 0.30, 0.16];
    assert!((0..3).all(|i| close(p[i], e[i], 1e-3)), "corrected pixel (B5b): got {p:?}, expected {e:?}");
}
