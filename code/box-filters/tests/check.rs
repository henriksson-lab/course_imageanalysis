// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p box-filters

use box_filters::*;
use imgcourse::{samples, Image};

/// The 5 × 5 image of exercise B3. The exercise counts from 1, the code from 0: the window
/// "centred on (3, 3)" is centred on (x, y) = (2, 2).
fn b3_image() -> Image<f32> {
    Image::from_rows(&[
        [2., 0., 1., 3., 1.],
        [1., 4., 2., 0., 2.],
        [3., 1., 5., 2., 0.],
        [0., 2., 1., 4., 3.],
        [1., 3., 0., 2., 5.],
    ])
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_running_sum_b1() {
    let f = [4., 8., 6., 2., 0., 4., 10., 6., 8., 2.];
    let cases = [
        (1, [6., 6., 16. / 3., 8. / 3., 2., 14. / 3., 20. / 3., 8., 16. / 3., 5.]),
        (2, [6., 5., 4., 4., 4.4, 4.4, 5.6, 6., 6.5, 16. / 3.]),
    ];
    for (r, expected) in cases {
        let got = box_mean_1d(&f, r);
        for i in 0..10 {
            assert!(close(got[i], expected[i], 1e-4), "width {}, position {} (from 1): got {}, expected {:.3}", 2 * r + 1, i + 1, got[i], expected[i]);
        }
    }
}

#[test]
fn c2_box_mean_equals_direct_clipped_mean() {
    let img = samples::cell().crop(30, 40, 23, 17);
    for r in [0, 1, 3, 12] {
        let out = box_mean(&img, r);
        for (x, y) in img.coords() {
            let (mut sum, mut n) = (0.0, 0);
            for yy in y.saturating_sub(r)..=(y + r).min(16) {
                for xx in x.saturating_sub(r)..=(x + r).min(22) {
                    sum += img[(xx, yy)];
                    n += 1;
                }
            }
            let e = sum / n as f32;
            assert!(close(out[(x, y)], e, 1e-5), "r = {r}, pixel ({x}, {y}): got {}, expected {e}", out[(x, y)]);
        }
    }
}

#[test]
fn c3_local_statistics_b3() {
    let (m, s) = local_mean_sd(&b3_image(), 1);
    // B3(a): centred on (3, 3) of the exercise; B3(c): on (4, 4)
    for ((x, y), em, es) in [((2, 2), 21. / 9., 2.444_f32.sqrt()), ((3, 3), 22. / 9., (84. / 9. - (22.0f32 / 9.).powi(2)).sqrt())] {
        assert!(close(m[(x, y)], em, 1e-4), "mean at ({}, {}) of the exercise: got {}, expected {em:.3}", x + 1, y + 1, m[(x, y)]);
        assert!(close(s[(x, y)], es, 1e-3), "SD at ({}, {}) of the exercise: got {}, expected {es:.3}", x + 1, y + 1, s[(x, y)]);
    }
}

#[test]
fn c4_normalisation_ignores_gain_and_offset_b7() {
    let img = samples::cell().crop(100, 100, 40, 40);
    let brighter = img.map(|v| 2.0 * v + 0.3);
    let a = normalise(&img, 5, 0.0);
    let b = normalise(&brighter, 5, 0.0);
    for (x, y) in img.coords() {
        assert!(close(a[(x, y)], b[(x, y)], 1e-2), "pixel ({x}, {y}): {} for the image, {} for 2·image + 0.3", a[(x, y)], b[(x, y)]);
    }
}

#[test]
fn c5_repeated_boxes_add_variances_b2() {
    // One bright point in the middle: after the passes, the image is the filter's weights.
    let mut point = Image::new(41, 41, 0.0f32);
    point[(20, 20)] = 1.0;
    // B2(a): two passes of width 3 give (1, 2, 3, 2, 1)/9 along each axis.
    let tent = repeated_box(&point, 1, 2);
    let w = [1., 2., 3., 2., 1.].map(|v: f32| v / 9.0);
    for i in 0..5 {
        let e = w[i] * w[2]; // row through the centre: weight across times weight of the centre row
        assert!(close(tent[(18 + i, 20)], e, 1e-6), "two passes of width 3, pixel {i} of the centre row: got {}, expected {e}", tent[(18 + i, 20)]);
    }
    // B2(b): width 5, n passes: variance n · (25 - 1)/12 = 2n along x.
    for n in 1..=3 {
        let out = repeated_box(&point, 2, n);
        let var: f32 = out.coords().map(|(x, y)| out[(x, y)] * (x as f32 - 20.0).powi(2)).sum();
        assert!(close(var, 2.0 * n as f32, 1e-3), "{n} passes of width 5: variance {var}, expected {}", 2 * n);
    }
}
