// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day33 pixelclass::
//
// Classes: 0 = background (B), 1 = nucleus (N).

use imgcourse::Image;
use day33::pixelclass::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The 5 × 5 patch of B1.
fn b1() -> Image<f32> {
    Image::from_rows(&[
        [10., 12., 14., 30., 60.],
        [12., 15., 40., 80., 90.],
        [14., 50., 90., 95., 92.],
        [40., 85., 96., 98., 94.],
        [70., 92., 97., 99., 96.],
    ])
}

#[test]
fn c1_features_b1() {
    let [b, g, l] = pixel_features(&b1(), 2, 2);
    assert!(close(b, 72.11, 0.01), "B1 (a): box mean at the centre: got {b}, expected 649/9 = 72.1");
    assert!(close(g, 35.92, 0.01), "B1 (a): gradient magnitude: got {g}, expected √(22.5² + 28²) = 35.9");
    assert!(close(l, -79.0, 1e-3), "B1 (a): Laplacian: got {l}, expected −79");
    // B1 (c): row 1, column 1 (value 15) is (x, y) = (1, 1)
    let [b, g, l] = pixel_features(&b1(), 1, 1);
    assert!(close(b, 28.56, 0.01), "B1 (c): box mean: got {b}, expected 257/9 = 28.6");
    assert!(close(g, 23.6, 0.05), "B1 (c): gradient magnitude: got {g}, expected 23.6");
    assert!(close(l, 54.0, 1e-3), "B1 (c): Laplacian: got {l}, expected 54");
}

#[test]
fn c2_stack() {
    let flat = Image::new(20, 20, 0.4);
    let stack = feature_stack(&flat, &[1.0, 3.0]);
    assert_eq!(stack.len(), 6, "2 scales × 3 channels: got {} channels", stack.len());
    for (c, ch) in stack.iter().enumerate() {
        let expected = if c % 3 == 0 { 0.4 } else { 0.0 };
        assert!(ch.pixels().iter().all(|&v| close(v, expected, 1e-5)), "a flat image: channel {c} should be {expected} everywhere");
    }
    // A bright spot: the blob channel (σ² ∇²L) is negative at its centre, the edge channel 0.
    let mut spot = Image::new(21, 21, 0.0);
    spot[(10, 10)] = 1.0;
    let stack = feature_stack(&spot, &[2.0]);
    assert!(stack[2][(10, 10)] < 0.0, "blob channel at a bright spot: got {}, expected negative", stack[2][(10, 10)]);
    assert!(stack[1][(10, 10)].abs() < 1e-6, "edge channel at the centre of a symmetric spot: got {}, expected 0", stack[1][(10, 10)]);
    assert!(stack[1][(12, 10)] > 0.0, "edge channel beside the spot: got {}, expected positive", stack[1][(12, 10)]);
}

#[test]
fn c3_weights_b2() {
    let w = class_weights(&[400, 20]);
    assert!(close(w[0], 0.525, 1e-5) && close(w[1], 10.5, 1e-4), "B2 (b): weights for 400 B and 20 N: got {w:?}, expected [0.525, 10.5]");
    let unweighted = weighted_shares(&[8, 2], &[1.0, 1.0]);
    assert!(close(unweighted[1], 0.2, 1e-6), "B2 (a): leaf with 8 B, 2 N: P(N) = 0.2, got {}", unweighted[1]);
    let weighted = weighted_shares(&[8, 2], &w);
    assert!(close(weighted[1], 0.833, 1e-3), "B2 (b): the same leaf, balanced: P(N) = 0.83, got {}", weighted[1]);
}

#[test]
fn c4_confusion_b3() {
    let truth = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    let predicted = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0];
    let m = confusion(&truth, &predicted, 2);
    assert_eq!(m, vec![vec![9, 2], vec![1, 4]], "B3 (a): rows = truth (B, N), columns = prediction: TN 9, FP 2 / FN 1, TP 4");
    assert!(close(dice(&m, 1), 8.0 / 11.0, 1e-5), "B3 (a): Dice of N: got {}, expected 0.727", dice(&m, 1));
    assert!(close(dice(&m, 0), 18.0 / 21.0, 1e-5), "B3 (a): Dice of B: got {}, expected 0.857", dice(&m, 0));
}
