// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day45 ratiometric::

use imgcourse::Image;
use day45::ratiometric::sim;
use day45::ratiometric::*;

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

#[test]
fn c1_ratio_image_b1() {
    // background 50 in both channels; a bright pixel, a dim pixel, and a background pixel
    let a = Image::from_rows(&[[150.0, 60.0, 52.0]]);
    let d = Image::from_rows(&[[450.0, 90.0, 47.0]]);
    let r = ratio_image(&a, &d, 50.0, 50.0, 20.0);
    assert!(close(r[(0, 0)], 0.25, 1e-6) && close(r[(1, 0)], 0.25, 1e-6), "B1, background subtracted: got {} and {}, expected 0.25 and 0.25", r[(0, 0)], r[(1, 0)]);
    assert!(r[(2, 0)].is_nan(), "a background pixel (sum below min_sum) should be NaN, got {}", r[(2, 0)]);
    let raw = ratio_image(&a, &d, 0.0, 0.0, 20.0);
    assert!(close(raw[(0, 0)], 0.333, 1e-3) && close(raw[(1, 0)], 0.667, 1e-3), "B1, without subtraction: got {} and {}, expected 0.333 and 0.667", raw[(0, 0)], raw[(1, 0)]);
}

#[test]
fn c2_ratio_of_sums_b4() {
    let a = Image::from_rows(&[[10.0, 50.0, 100.0, 200.0]]);
    let d = Image::from_rows(&[[5.0, 50.0, 200.0, 400.0]]);
    let all = Image::new(4, 1, true);
    let rs = ratio_of_sums(&a, &d, &all);
    let mr = mean_of_ratios(&a, &d, &all);
    assert!(close(rs, 360.0 / 655.0, 1e-5), "B4(a): ratio of sums got {rs}, expected 0.550");
    assert!(close(mr, 1.0, 1e-6), "B4(a): mean of ratios got {mr}, expected 1.00");
    let no_first = Image::from_rows(&[[false, true, true, true]]);
    let mr = mean_of_ratios(&a, &d, &no_first);
    assert!(close(mr, 2.0 / 3.0, 1e-6), "only the masked pixels: got {mr}, expected 0.667");
}

#[test]
fn c3_calibrate_b5() {
    for (r, c) in [(1.5, 200.0), (2.3, 1800.0), (2.4, 3800.0), (1.45, 181.0), (1.55, 221.0)] {
        let got = calibrate(r, 200.0, 0.5, 2.5);
        assert!(close(got, c, 1.0), "B5: R = {r}: got {got:.0} nM, expected {c}");
    }
    let back = calibrate(sim::true_ratio(700.0), sim::KD, sim::R_MIN, sim::R_MAX);
    assert!(close(back, 700.0, 0.5), "calibrate should undo the sensor curve: got {back}");
}

#[test]
fn c4_bin() {
    let img = Image::from_fn(9, 8, |x, y| (x + 10 * y) as f32);
    let b = bin(&img, 4);
    assert_eq!((b.width(), b.height()), (2, 2), "9 × 8 binned by 4 is 2 × 2 (the last column is dropped)");
    let first: f32 = (0..4).flat_map(|y| (0..4).map(move |x| (x + 10 * y) as f32)).sum();
    assert!(close(b[(0, 0)], first, 1e-3), "block (0, 0): got {}, expected {first}", b[(0, 0)]);
    assert!(close(b[(1, 1)], first + 16.0 * 4.0 + 16.0 * 40.0, 1e-3), "block (1, 1): got {}", b[(1, 1)]);
}

#[test]
fn c5_intensity_modulated() {
    let ratio = Image::from_rows(&[[0.5, 2.5, 1.5, f32::NAN]]);
    let intensity = Image::from_rows(&[[100.0, 200.0, 50.0, 100.0]]);
    let img = intensity_modulated(&ratio, &intensity, 0.5, 2.5, 100.0);
    assert_eq!(img[(0, 0)], sim::colour_map(0.0), "r = r_lo at full intensity: the first colour of the map");
    assert_eq!(img[(1, 0)], sim::colour_map(1.0), "r = r_hi, intensity above i_max: the last colour, not brighter");
    let half = sim::colour_map(0.5).map(|c| 0.5 * c);
    for k in 0..3 {
        assert!(close(img[(2, 0)][k], half[k], 1e-6), "half the intensity: half the colour, got {:?}, expected {half:?}", img[(2, 0)]);
    }
    assert_eq!(img[(3, 0)], [0.0; 3], "no ratio (NaN): black");
}
