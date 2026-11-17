// Checks for the coding exercises, using the numbers of the pen-and-paper exercises (A1 to A4, B7).
// Run: cargo test -p day24 descriptors::

use day24::descriptors::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const A1: [(f32, f32); 4] = [(2.0, 10.0), (1.0, 50.0), (3.0, 100.0), (2.0, 355.0)];

#[test]
fn c1_cell_histogram_a1() {
    let got = cell_histogram(&A1);
    assert_eq!(got, [4.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], "A1: got {got:?}, expected [4, 1, 3, 0, 0, 0, 0, 0]");
    let turned: Vec<(f32, f32)> = A1.iter().map(|&(m, d)| (m, d + 90.0)).collect();
    let got = cell_histogram(&turned);
    assert_eq!(got, [0.0, 0.0, 4.0, 1.0, 3.0, 0.0, 0.0, 0.0], "A1 turned by 90°: got {got:?}, expected [0, 0, 4, 1, 3, 0, 0, 0]");
}

#[test]
fn c2_normalise_clip_a2() {
    // A2: twice as bright doubles the magnitudes; normalised, both give [0.78, 0.20, 0.59, 0, ...].
    let a = normalise_clip(&[4.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    let b = normalise_clip(&[8.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    for (i, e) in [0.784, 0.196, 0.588].iter().enumerate() {
        assert!(close(a[i], *e, 1e-3) && close(b[i], *e, 1e-3), "A2 entry {i}: got {} and {}, expected {e}", a[i], b[i]);
    }
    // Clipping at 0.2: one large entry no longer dominates.
    let v = normalise_clip(&[10.0, 1.0, 1.0, 1.0], 0.2);
    let length: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    assert!(close(length, 1.0, 1e-4), "after clipping, normalise again: length {length}, expected 1");
    // Normalised: 0.985 and 3 × 0.0985; clipped: 0.2 and 3 × 0.0985; normalised again: 0.761, 0.375.
    let expected = 0.2 / (0.04f32 + 3.0 / 103.0).sqrt();
    assert!(close(v[0], expected, 1e-3), "[10, 1, 1, 1] clipped at 0.2: got {v:?}, expected the first entry {expected}");
}

#[test]
fn c3_sample_patch_frame() {
    // θ = 0 and 12σ/n = 1: the patch is the image itself, shifted by half a pixel.
    let img = Image::from_fn(20, 20, |x, y| (x + 20 * y) as f32);
    let p = sample_patch(&img, 10.0, 10.0, 4.0 / 3.0, 0.0, 16);
    assert!(close(p[(0, 0)], 52.5, 1e-3), "θ = 0: sample (0, 0) lies at (2.5, 2.5): got {}, expected 52.5", p[(0, 0)]);
    // θ = 90° (counter-clockwise): the frame's x axis points up on the screen.
    let p = sample_patch(&img, 10.0, 10.0, 4.0 / 3.0, 90.0, 16);
    // Sample (12, 0): u = 4.5, v = −7.5 in the frame, so on the screen (10 + v, 10 − u) = (2.5, 5.5).
    let expected = 10.0 + 20.0 * 5.5;
    let got = p[(12, 0)];
    assert!(close(got, expected - 7.5, 1e-2), "θ = 90°, sample (12, 0): got {got}, expected {}", expected - 7.5);
}

#[test]
fn c3_normalised_patch_ignores_gain_and_offset() {
    let patch = samples::cell().crop(100, 100, 16, 16);
    let a = normalised_patch(&patch);
    let b = normalised_patch(&patch.map(|v| 3.0 * v + 0.2));
    assert!(distance(&a, &b) < 1e-4, "gain 3 and offset 0.2 must not change the normalised patch (distance {})", distance(&a, &b));
    let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
    assert!(close(mean, 0.0, 1e-5), "mean of the normalised patch: got {mean}, expected 0");
}

#[test]
fn c4_gradient_descriptor_invariance_b7() {
    let patch = samples::cell().crop(120, 90, 16, 16);
    let d = gradient_descriptor(&patch);
    assert_eq!(d.len(), 128, "4 × 4 cells × 8 bins = 128 numbers, got {}", d.len());
    let length: f32 = d.iter().map(|a| a * a).sum::<f32>().sqrt();
    assert!(close(length, 1.0, 1e-4), "the descriptor must have length 1, got {length}");
    // B7: I → aI + b with a > 0 does not change the descriptor.
    let e = gradient_descriptor(&patch.map(|v| 2.5 * v - 0.3));
    assert!(distance(&d, &e) < 1e-4, "B7: gain and offset changed the descriptor (distance {})", distance(&d, &e));
    // A vertical step edge, bright on the right: every gradient points right (0°), so only
    // bin 0 of each cell can be non-zero.
    let step = samples::step(16, 16, 0.0, 1.0);
    let s = gradient_descriptor(&step);
    for (i, v) in s.iter().enumerate() {
        assert!(i % 8 == 0 || *v == 0.0, "step edge: entry {i} (bin {}) is {v}, expected only bin 0 to be used", i % 8);
    }
}

#[test]
fn c5_hamming_a3() {
    let a = 0b1011_0010_1110_0001u64;
    let b = 0b1001_0110_1100_0011u64;
    assert_eq!(hamming(&[a], &[b]), 4, "A3: the Hamming distance is 4");
    assert_eq!(hamming(&[a, u64::MAX], &[a, 0]), 64, "two words: 0 + 64 differing bits");
}

#[test]
fn c5_binary_descriptor_a4() {
    // A4: rows from the top; pairs given as (row, column) in the exercise, (x, y) = (column, row) here.
    let patch = Image::from_rows(&[[12.0, 40.0, 25.0], [30.0, 55.0, 18.0], [60.0, 35.0, 47.0]]);
    let pairs = [((0, 0), (1, 1)), ((1, 0), (0, 2)), ((2, 2), (2, 0)), ((0, 1), (1, 2))];
    let bits = binary_descriptor(&patch, &pairs);
    assert_eq!(bits, vec![0b1011], "A4(a): bits 1, 1, 0, 1 (bit 0 first) = 0b1011, got {:#b}", bits[0]);
    let squared = binary_descriptor(&patch.map(|v| v * v), &pairs);
    assert_eq!(squared, bits, "A4(b): squaring keeps the order, the bits must not change");
    let pairs300: Vec<_> = (0..300).map(|i| if i % 2 == 0 { ((0, 0), (1, 1)) } else { ((1, 1), (0, 0)) }).collect();
    let long = binary_descriptor(&patch, &pairs300);
    assert_eq!(long.len(), 5, "300 bits need 5 words of 64, got {}", long.len());
    assert_eq!(long[0], 0x5555_5555_5555_5555, "bit i is bit i % 64 of word i / 64");
}
