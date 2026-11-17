// Checks for the coding exercises of "Image-to-image translation and hallucination", with the
// numbers of the pen-and-paper exercises. Run: cargo test -p gans image_translation::

use gans::image_translation::*;
use imgcourse::Rng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// The cells of B1: four cells at x = 0.7, two with the marker off (0.6), two on (0.9).
fn b1() -> Vec<Pair> {
    [0.6, 0.6, 0.9, 0.9].iter().map(|&y| Pair { x: 0.7, y, on: y > 0.7 }).collect()
}

#[test]
fn c1_mean_b1() {
    let m = cond_mean(&b1(), 0.7, 0.04);
    assert!(close(m, 0.75, 1e-12), "B1: the mean of 0.6, 0.6, 0.9, 0.9 is 0.75; got {m}");
    let far = vec![Pair { x: 0.2, y: 0.3, on: false }, Pair { x: 0.8, y: 0.9, on: false }];
    let m = cond_mean(&far, 0.21, 0.04);
    assert!(close(m, 0.3, 1e-6), "near x = 0.2 only the first pair counts; got {m}");
}

#[test]
fn c2_median_b1() {
    let m = cond_median(&b1(), 0.7, 0.04);
    assert!(close(m, 0.6, 1e-12), "B1: the first value whose cumulative weight reaches half is 0.6; got {m}");
    let three: Vec<Pair> = [0.6, 0.9, 0.9].iter().map(|&y| Pair { x: 0.7, y, on: false }).collect();
    assert!(close(cond_median(&three, 0.7, 0.04), 0.9, 1e-12), "median of 0.6, 0.9, 0.9 is 0.9");
}

#[test]
fn c3_samples_are_training_values() {
    let mut rng = Rng::new(4);
    let mut on = 0;
    for _ in 0..2000 {
        let y = cond_sample(&b1(), 0.7, 0.04, &mut rng);
        assert!(y == 0.6 || y == 0.9, "a sample must be one of the training targets; got {y}");
        if y == 0.9 {
            on += 1;
        }
    }
    assert!((900..1100).contains(&on), "about half the samples should be 0.9; got {on} of 2000");
}

#[test]
fn c4_quantile_map_b4() {
    let (xs, ys) = ([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]);
    assert_eq!(quantile_map(&xs, &ys, 2.5, false), 30.0, "B4: 2 of 4 inputs below 2.5: the target with rank 2 (from 0)");
    assert_eq!(quantile_map(&xs, &ys, 1.0, false), 10.0, "the smallest input goes to the smallest target");
    assert_eq!(quantile_map(&xs, &ys, 1.0, true), 40.0, "decreasing: the smallest input goes to the largest target");
}

#[test]
fn c5_scores_b2() {
    let e = rmse(&[0.75, 0.75], &[0.6, 0.9]);
    assert!(close(e, 0.15, 1e-12), "B2: the mean's error is 0.15; got {e}");
    let e = rmse(&[0.6, 0.9, 0.9, 0.6], &[0.6, 0.6, 0.9, 0.9]);
    assert!(close(e, 0.2121, 1e-4), "B2: random branches: half right, half 0.3 off: 0.212; got {e}");
    assert!(is_real(0.7, 0.48, 0.3) && is_real(0.7, 0.78, 0.3) && !is_real(0.7, 0.63, 0.3), "B2: 0.48 and 0.78 are real at x = 0.7 (gap 0.3), 0.63 is not");
    assert!(!is_real(0.3, 0.62, 0.3), "below x = 0.5 there is no marker branch");
}
