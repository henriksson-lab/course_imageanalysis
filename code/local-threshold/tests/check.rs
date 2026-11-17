// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p local-threshold
// The checks use grey levels 0–255 as in the exercises; the functions do not mind.

use imgcourse::Image;
use local_threshold::*;

/// B2: a 1 × 10 profile on a rising background.
fn b2() -> Image<f32> {
    Image::from_rows(&[[100., 110., 80., 130., 140., 150., 160., 130., 180., 190.]])
}

fn objects(mask: &Image<bool>) -> Vec<usize> {
    (0..mask.width()).filter(|&x| mask[(x, 0)]).collect()
}

#[test]
fn c1_local_mean_b2() {
    let (m, _) = local_mean_std(&b2(), 1);
    let expected = [103.3, 96.7, 106.7, 116.7, 140.0, 150.0, 146.7, 156.7, 166.7, 186.7];
    for (x, e) in expected.iter().enumerate() {
        assert!((m[(x, 0)] - e).abs() < 0.05, "B2 (b): mean at position {}: got {:.1}, expected {e}", x + 1, m[(x, 0)]);
    }
}

#[test]
fn c1_local_std_b2_b7() {
    let (_, s) = local_mean_std(&b2(), 1);
    let expected = [4.7, 12.5, 20.6, 26.2, 8.2, 8.2, 12.5, 20.6, 26.2, 4.7];
    for (x, e) in expected.iter().enumerate() {
        assert!((s[(x, 0)] - e).abs() < 0.06, "B2 (c): standard deviation at position {}: got {:.2}, expected {e}", x + 1, s[(x, 0)]);
    }
    assert!((s[(0, 0)].powi(2) - 22.2).abs() < 0.05, "B7: variance of (100, 100, 110): got {:.2}, expected 22.2", s[(0, 0)].powi(2));
}

#[test]
fn c1_window_in_2d() {
    // a 5 × 5 image with a single 9 in the middle: the 3 × 3 mean is 1 next to it, 0 far away
    let mut img = Image::new(5, 5, 0.0);
    img[(2, 2)] = 9.0;
    let (m, _) = local_mean_std(&img, 1);
    assert!((m[(1, 1)] - 1.0).abs() < 1e-5 && m[(0, 4)].abs() < 1e-5, "3 × 3 mean next to the 9: got {}, far away: {}", m[(1, 1)], m[(0, 4)]);
}

#[test]
fn c2_sauvola_b3() {
    let flat = rule_threshold(Rule::Sauvola { k: 0.3, range: 128.0 }, 200.0, 6.0);
    let edge = rule_threshold(Rule::Sauvola { k: 0.3, range: 128.0 }, 150.0, 50.0);
    assert!((flat - 142.8).abs() < 0.05, "B3 (b): Sauvola in flat background: got {flat:.1}, expected 142.8");
    assert!((edge - 122.6).abs() < 0.05, "B3 (c): Sauvola at an edge: got {edge:.1}, expected 122.6");
    let nib = rule_threshold(Rule::Niblack(-0.2), 200.0, 6.0);
    assert!((nib - 198.8).abs() < 1e-3, "B3 (a): Niblack in flat background: got {nib}, expected 198.8");
    assert!((rule_threshold(Rule::MeanC(10.0), 103.3, 4.7) - 93.3).abs() < 1e-3, "mean − C: 103.3 − 10");
}

#[test]
fn c3_mean_c_b2() {
    let mask = local_threshold(&b2(), 1, Rule::MeanC(10.0));
    // positions 3 and 8, counted from 1 in the exercise
    assert_eq!(objects(&mask), vec![2, 7], "B2 (b): mean − C should find positions 3 and 8 (x = 2 and 7)");
}

#[test]
fn c3_niblack_b2() {
    let mask = local_threshold(&b2(), 1, Rule::Niblack(-0.2));
    assert_eq!(objects(&mask), vec![0, 2, 7], "B2 (c): Niblack should call positions 1, 3 and 8 object (x = 0, 2, 7)");
}

/// B5: the 6 × 6 grid.
fn b5() -> Image<f32> {
    Image::from_rows(&[
        [10., 10., 10., 10., 10., 10.],
        [10., 90., 60., 10., 10., 10.],
        [10., 10., 60., 10., 70., 10.],
        [10., 10., 10., 60., 10., 10.],
        [10., 10., 10., 60., 10., 50.],
        [10., 70., 10., 10., 10., 60.],
    ])
}

fn kept(mask: &Image<bool>) -> Vec<(usize, usize)> {
    mask.coords().filter(|&p| mask[p]).collect()
}

#[test]
fn c4_hysteresis_4_b5() {
    let got = kept(&hysteresis(&b5(), 50.0, 80.0, false));
    // (x, y) from 0: rows 2–3, columns 2–3 of the exercise, counted from 1
    assert_eq!(got, vec![(1, 1), (2, 1), (2, 2)], "B5 (b): 4-connectivity keeps 3 pixels");
}

#[test]
fn c4_hysteresis_8_b5() {
    let got = kept(&hysteresis(&b5(), 50.0, 80.0, true));
    assert_eq!(got, vec![(1, 1), (2, 1), (2, 2), (4, 2), (3, 3), (3, 4)], "B5 (c): 8-connectivity keeps 6 pixels");
}

#[test]
fn c4_single_thresholds_b5() {
    // with low = high the result is a plain threshold: B5 (d)
    assert_eq!(kept(&hysteresis(&b5(), 80.0, 80.0, true)).len(), 1, "B5 (d): threshold 80 keeps 1 pixel");
    assert_eq!(kept(&hysteresis(&b5(), 50.0, 50.0, true)).len(), 8, "B5 (d): threshold 50 keeps 8 pixels");
}
