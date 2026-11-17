// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p thresholding histogram::

use thresholding::histogram::*;
use imgcourse::Image;

/// The 5 × 5 image of B1, grey levels 0–7. Level `i` is stored as `i / 8`, so that it lands
/// in bin `i` of a histogram with 8 bins.
fn b1() -> Image<f32> {
    let levels = [[0., 1., 1., 2., 1.], [1., 6., 7., 6., 2.], [1., 7., 5., 6., 1.], [2., 6., 6., 3., 0.], [1., 2., 1., 0., 4.]];
    Image::from_rows(&levels).map(|v| v / 8.0)
}

fn count(mask: &Image<bool>) -> usize {
    mask.pixels().iter().filter(|&&b| b).count()
}

#[test]
fn c1_histogram_b1() {
    let h = histogram(&b1(), 8);
    assert_eq!(h, vec![3, 8, 4, 1, 1, 1, 5, 2], "B1 (a): the counts for grey levels 0–7");
}

#[test]
fn c1_eight_bit_levels() {
    // 8-bit grey levels k / 255 land in bin k of 256 bins, 0 and 255 included
    let img = Image::from_fn(256, 1, |x, _| x as f32 / 255.0);
    let h = histogram(&img, 256);
    assert!(h.iter().all(|&c| c == 1), "each of the 256 grey levels once: got {:?}", &h[..8]);
}

#[test]
fn c2_cumulative_b1() {
    let p = cumulative(&[3, 8, 4, 1, 1, 1, 5, 2]);
    let expected = [0.12, 0.44, 0.60, 0.64, 0.68, 0.72, 0.92, 1.00];
    for (t, (g, e)) in p.iter().zip(expected).enumerate() {
        assert!((g - e).abs() < 1e-5, "P({t}): got {g}, expected {e}");
    }
}

#[test]
fn c3_threshold_b1() {
    let img = b1();
    let n3 = count(&threshold(&img, 8, 3));
    let n4 = count(&threshold(&img, 8, 4));
    let n1 = count(&threshold(&img, 8, 1));
    let n2 = count(&threshold(&img, 8, 2));
    assert_eq!((n3, n4), (9, 8), "B1 (b): object pixels at t = 3 and t = 4");
    assert_eq!((n1, n2), (14, 10), "B1 (c): object pixels at t = 1 and t = 2");
    // the one pixel that changes between t = 3 and t = 4 is the 4 in the bottom-right corner
    let m = threshold(&img, 8, 3);
    assert!(m[(4, 4)] && !threshold(&img, 8, 4)[(4, 4)], "B1 (b): the 4 at row 5, column 5 should change");
}

#[test]
fn c4_percentile_b1() {
    // keep the brightest 8 % of 25 pixels = 2 pixels: the two 7s
    let t = percentile_threshold(&[3, 8, 4, 1, 1, 1, 5, 2], 0.08);
    assert_eq!(t, 6, "brightest 8 % of B1: got t = {t}, expected 6");
}

#[test]
fn c4_percentile_b4() {
    // A histogram of 1000 pixels with P(100) = 0.950, P(110) = 0.968, P(120) = 0.980,
    // P(130) = 0.988, P(255) = 1 (as in B4).
    let mut h = vec![0; 256];
    h[100] = 950;
    h[110] = 18;
    h[120] = 12;
    h[130] = 8;
    h[200] = 12;
    let t = percentile_threshold(&h, 0.02);
    assert_eq!(t, 120, "B4 (a): brightest 2 %: got t = {t}, expected 120");
}

#[test]
fn c5_mean_variance_b8() {
    let (m, v) = mean_variance(&[3, 8, 4, 1, 1, 1, 5, 2]);
    assert!((m - 2.88).abs() < 1e-4, "B8: mean: got {m}, expected 2.88");
    assert!((v - 5.7856).abs() < 1e-3, "B8: variance: got {v}, expected 5.79");
}
