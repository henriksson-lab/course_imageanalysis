// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day38 vae_gen::

use imgcourse::Image;
use day38::vae_gen::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn assert_pixels(img: &Image<f32>, expected: &[f32], what: &str) {
    let got = img.pixels();
    assert!(got.len() == expected.len() && got.iter().zip(expected).all(|(a, b)| close(*a, *b, 1e-5)), "{what}: got {got:?}, expected {expected:?}");
}

#[test]
fn c1_latent_walk_b1() {
    let walk = latent_walk(&bar_decoder, [0.0, 0.0], [2.0, 0.0], 5);
    assert_eq!(walk.len(), 5, "5 steps from z = 0 to z = 2");
    let expected = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]];
    for (k, (img, e)) in walk.iter().zip(expected).enumerate() {
        assert_pixels(img, &e, &format!("B1, step {k} (z = {})", 0.5 * k as f32));
    }
}

#[test]
fn c1_pixel_fade_b2() {
    let (a, b) = (bar_decoder([0.0, 0.0]), bar_decoder([2.0, 0.0]));
    let fade = pixel_fade(&a, &b, 3);
    assert_eq!(fade.len(), 3);
    assert_pixels(&fade[0], &[1.0, 0.0, 0.0], "B2, fade at t = 0");
    assert_pixels(&fade[1], &[0.5, 0.0, 0.5], "B2, fade at t = 0.5 (two half bars)");
    assert_pixels(&fade[2], &[0.0, 0.0, 1.0], "B2, fade at t = 1");
}

#[test]
fn c2_fraction_within_b4() {
    for (r, expected) in [(1.0, 0.393), (2.0, 0.865), (3.0, 0.989)] {
        let got = fraction_within(r, 100_000, 1);
        assert!(close(got, expected, 0.01), "B4, radius {r}: got {got}, expected 1 − e^(−r²/2) = {expected}");
    }
}

#[test]
fn c3_encode_by_search_finds_the_code() {
    let decode = spot_decoder(24);
    let img = decode([0.5, -1.0]);
    let (z, err) = encode_by_search(&img, &decode, -3.0, 3.0, 61); // steps of 0.1
    assert!(close(z[0], 0.5, 1e-4) && close(z[1], -1.0, 1e-4), "the spot drawn at z = (0.5, −1): found z = {z:?}");
    assert!(err < 1e-8, "its own image is rebuilt exactly: error {err}");
    let (_, err) = encode_by_search(&Image::new(24, 24, 1.0), &decode, -3.0, 3.0, 13);
    assert!(err > 0.5, "a white image cannot be drawn by the spot decoder: error {err} should be large");
}

#[test]
fn c4_threshold_b6() {
    let small = threshold(&[0.3, 0.1, 0.4, 0.2], 0.5);
    assert!(close(small, 0.2, 1e-6), "half of 0.1, 0.2, 0.3, 0.4 is at or below 0.2, got {small}");
    // 2000 normal errors, threshold at 99 %: 20 false alarms (B6).
    let errors: Vec<f32> = (1..=2000).map(|k| k as f32 / 2000.0 * 0.05).collect();
    let t = threshold(&errors, 0.99);
    let flagged = errors.iter().filter(|&&e| e > t).count();
    assert_eq!(flagged, 20, "B6: with the 99 % threshold {t}, 1 % of 2000 normal images should be flagged, got {flagged}");
}
