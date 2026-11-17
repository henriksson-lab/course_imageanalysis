// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day42 colour_display::

use day42::colour_display::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_position_b2() {
    let t = position(-3.0, -7.0, 7.0);
    assert!(close(t, 0.286, 1e-3), "−3 in the range −7..7 (B2b): got t = {t}, expected 0.286");
    let t = position(7.0, -7.0, 7.0);
    assert!(close(t, 1.0, 1e-6), "7 in −7..7: got t = {t}, expected 1");
    let t = position(2.0, -3.0, 7.0);
    assert!(close(t, 0.5, 1e-6), "2 in −3..7 (B2a, the white centre): got t = {t}, expected 0.5");
    let t = position(9.0, -3.0, 7.0);
    assert!(close(t, 1.0, 1e-6), "a value above the range: got t = {t}, expected 1 (clipped)");
}

#[test]
fn c1_colour_map_interpolates() {
    let anchors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let img = Image::from_rows(&[[0.0f32, 0.25, 0.5, 0.75, 1.0, 2.0]]);
    let out = colour_map(&img, 0.0, 1.0, &anchors);
    let expected = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.5], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    for x in 0..6 {
        let got = out[(x, 0)];
        assert!((0..3).all(|k| close(got[k], expected[x][k], 1e-5)), "value {}: got {got:?}, expected {:?}", img[(x, 0)], expected[x]);
    }
}

#[test]
fn c2_lightness() {
    assert!(close(lightness([1.0; 3]), 100.0, 0.05), "white: got {}, expected 100", lightness([1.0; 3]));
    assert!(close(lightness([0.0; 3]), 0.0, 0.05), "black: got {}, expected 0", lightness([0.0; 3]));
    let l = lightness([0.4613; 3]); // 18 % of white's light
    assert!(close(l, 49.5, 0.1), "18 % grey: got {l}, expected 49.5");
    let (r, g) = (lightness([1.0, 0.0, 0.0]), lightness([0.0, 1.0, 0.0]));
    assert!(close(r, 53.2, 0.2) && close(g, 87.7, 0.2), "pure red and green: got {r} and {g}, expected 53.2 and 87.7");
}

#[test]
fn c2_monotone_b1() {
    let jet = [12.9, 54.6, 90.6, 67.0, 25.4];
    let viridis = [14.9, 33.1, 49.9, 66.5, 90.9];
    assert!(!is_monotone(&jet), "jet's profile (B1) is not monotone");
    assert!(is_monotone(&viridis), "viridis's profile (B1) is monotone");
    assert!(is_monotone(&[3.0, 2.0, 1.0]), "a decreasing profile is monotone too");
    assert!(!is_monotone(&[1.0, 1.0, 2.0]), "a flat step is not strictly monotone");
}

#[test]
fn c3_merge() {
    let a = Image::from_rows(&[[1.0f32, 0.0, 1.0, 0.5]]);
    let b = Image::from_rows(&[[0.0f32, 1.0, 1.0, 0.5]]);
    let m = merge(&[a, b], &[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]);
    let expected = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]];
    for x in 0..4 {
        let got = m[(x, 0)];
        assert!((0..3).all(|k| close(got[k], expected[x][k], 1e-6)), "pixel {x} (magenta + green): got {got:?}, expected {:?}", expected[x]);
    }
    let big = merge(&[Image::new(1, 1, 0.8), Image::new(1, 1, 0.7)], &[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    assert!(close(big[(0, 0)][0], 1.0, 1e-6), "0.8 + 0.7 in red: got {}, expected 1 (clipped)", big[(0, 0)][0]);
}

#[test]
fn c4_deutan_b3() {
    // Anchors chosen so that a = b = 0.5, the toy deuteranope of B3.
    let d = deutan_lms([1.0, 1.0, 1.0], [0.0, 0.5, 1.0]);
    let expected = [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!(close(d[i][j], expected[i][j], 1e-5), "row {}, column {}: got {}, expected {}", i + 1, j + 1, d[i][j], expected[i][j]);
        }
    }
    let apply = |v: [f32; 3]| d.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
    let (r, g) = (apply([0.8, 0.4, 0.2]), apply([0.8, 0.9, 0.2]));
    assert!((0..3).all(|k| close(r[k], g[k], 1e-6)), "B3(a): reddish → {r:?}, greenish → {g:?}, expected the same");
    // the anchors are kept, for any anchors
    let (w, q) = ([0.9, 0.8, 0.95], [0.05, 0.1, 0.9]);
    let d = deutan_lms(w, q);
    for v in [w, q] {
        let m = d[1][0] * v[0] + d[1][2] * v[2];
        assert!(close(m, v[1], 1e-5), "anchor {v:?}: M becomes {m}, expected {}", v[1]);
    }
}

#[test]
fn c5_factors_b4() {
    let f = stretch_factors([7.0, 1.0, 1.0], 1.0);
    let e = [0.655, 1.732, 1.732];
    assert!((0..3).all(|k| close(f[k], e[k], 1e-3)), "B4(c): got {f:?}, expected {e:?}");
    let f = stretch_factors([7.0, 1.0, 1.0], 0.0);
    assert!(f.iter().all(|&v| close(v, 1.0, 1e-6)), "amount 0: got {f:?}, expected (1, 1, 1)");
}

#[test]
fn c5_stretch_whitens() {
    // A correlated cloud: the covariance after a full stretch must be σ̄² I.
    let mut rng = imgcourse::Rng::new(7);
    let img = Image::from_fn(40, 40, |_, _| {
        let (a, b, c) = (rng.normal(), rng.normal(), rng.normal());
        [0.5 + 0.1 * a + 0.02 * b, 0.5 + 0.09 * a - 0.02 * b + 0.005 * c, 0.4 + 0.05 * a + 0.01 * c]
    });
    let cov = |img: &Image<[f32; 3]>| {
        let n = 1600.0;
        let mut mu = [0.0f32; 3];
        img.pixels().iter().for_each(|p| (0..3).for_each(|k| mu[k] += p[k] / n));
        let mut s = [[0.0f32; 3]; 3];
        for p in img.pixels() {
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += (p[i] - mu[i]) * (p[j] - mu[j]) / n;
                }
            }
        }
        (mu, s)
    };
    let (mu0, s0) = cov(&img);
    let (mu1, s1) = cov(&decorrelation_stretch(&img, 1.0));
    let mean_var = (s0[0][0] + s0[1][1] + s0[2][2]) / 3.0;
    for i in 0..3 {
        assert!(close(mu1[i], mu0[i], 1e-4), "the mean colour must stay: got {mu1:?}, expected {mu0:?}");
        for j in 0..3 {
            let e = if i == j { mean_var } else { 0.0 };
            assert!(close(s1[i][j], e, 0.02 * mean_var), "covariance after the stretch, ({i}, {j}): got {}, expected {e}", s1[i][j]);
        }
    }
}
