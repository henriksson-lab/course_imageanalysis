// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p training augmentation::

use imgcourse::{Image, Rng};
use training::augmentation::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn small() -> Image<f32> {
    Image::from_rows(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
}

#[test]
fn c1_flip_horizontal() {
    let f = flip_horizontal(&small());
    assert!(close(f[(0, 0)], 0.3, 1e-6) && close(f[(2, 1)], 0.4, 1e-6), "flip: got first row {:?}", f.row(0));
    let back = flip_horizontal(&f);
    assert!(back.pixels() == small().pixels(), "flipping twice should give the image back");
}

#[test]
fn c2_rotate_quarter_turn_clockwise() {
    let img = small();
    let r = rotate(&img, 90.0);
    // B2: a clockwise quarter turn takes the top-left corner to the top-right
    assert!(close(r[(2, 0)], 0.1, 1e-4), "rotate 90: pixel (2, 0) is {}, expected 0.1 (the old top-left)", r[(2, 0)]);
    assert!(close(r[(0, 0)], 0.7, 1e-4), "rotate 90: pixel (0, 0) is {}, expected 0.7 (the old bottom-left)", r[(0, 0)]);
    assert!(close(r[(1, 1)], 0.5, 1e-6), "the centre stays");
    let same = rotate(&img, 0.0);
    assert!(img.pixels().iter().zip(same.pixels()).all(|(a, b)| close(*a, *b, 1e-6)), "rotate by 0 should change nothing");
}

#[test]
fn c3_elastic() {
    let img = imgcourse::samples::cell().crop(0, 0, 64, 64);
    let none = elastic(&img, 0.0, 4.0, 1);
    assert!(img.pixels().iter().zip(none.pixels()).all(|(a, b)| close(*a, *b, 1e-5)), "alpha 0 should change nothing");
    let a = elastic(&img, 2.0, 4.0, 1);
    let b = elastic(&img, 2.0, 4.0, 1);
    assert!(a.pixels() == b.pixels(), "the same seed should give the same deformation");
    let flat = Image::new(32, 32, 0.4);
    let moved = elastic(&flat, 3.0, 4.0, 2);
    assert!(moved.pixels().iter().all(|v| close(*v, 0.4, 1e-5)), "a flat image stays flat under any deformation");
    let diff = img.pixels().iter().zip(a.pixels()).map(|(p, q)| (p - q).abs()).sum::<f32>() / 4096.0;
    assert!(diff > 0.005, "alpha 2 should move the picture (mean change only {diff})");
}

#[test]
fn c4_intensity_b6() {
    // B6: v = 0.25, gain 2, offset 0.1, gamma 0.5: 2 * 0.5 + 0.1 = 1.1
    let img = Image::new(2, 2, 0.25);
    let out = intensity(&img, 2.0, 0.1, 0.5);
    assert!(close(out[(1, 1)], 1.1, 1e-5), "B6: got {}, expected 1.1", out[(1, 1)]);
    let neg = intensity(&Image::new(1, 1, -0.2), 1.0, 0.0, 0.5);
    assert!(close(neg[(0, 0)], 0.0, 1e-6), "negative values should become 0 before the power, got {}", neg[(0, 0)]);
}

#[test]
fn c5_augment_off_changes_nothing() {
    let img = small();
    let off = Augment { flips: false, max_angle: 0.0, gain_range: 0.0, noise: 0.0 };
    let mut rng = Rng::new(1);
    let out = augment(&img, &off, &mut rng);
    assert!(img.pixels().iter().zip(out.pixels()).all(|(a, b)| close(*a, *b, 1e-6)), "augment with everything off should return the image");
}

#[test]
fn c5_augment_draws_in_order() {
    // the gain only: with seed 4 the first two uniform numbers are the angle's and the gain's
    let img = Image::new(2, 2, 0.5);
    let a = Augment { flips: false, max_angle: 0.0, gain_range: 0.4, noise: 0.0 };
    let mut rng = Rng::new(4);
    let out = augment(&img, &a, &mut rng);
    let mut r = Rng::new(4);
    let _angle = r.uniform();
    let gain = 1.0 + 0.4 * (2.0 * r.uniform() - 1.0);
    assert!(close(out[(0, 0)], 0.5 * gain, 1e-5), "gain: got {}, expected {} (draw the angle's number first, then the gain's)", out[(0, 0)], 0.5 * gain);
    let flips = Augment { flips: true, max_angle: 0.0, gain_range: 0.0, noise: 0.0 };
    let mut seen = std::collections::HashSet::new();
    let mut rng = Rng::new(9);
    for _ in 0..40 {
        let o = augment(&small(), &flips, &mut rng);
        seen.insert((o[(0, 0)] * 10.0).round() as i32);
    }
    assert_eq!(seen.len(), 4, "flips: the top-left pixel should come from all four corners, got {seen:?}");
}
