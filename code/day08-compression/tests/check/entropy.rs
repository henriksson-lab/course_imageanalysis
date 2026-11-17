// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day08 entropy::

use day08::entropy::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_histogram_counts_every_pixel() {
    let img = Image::from_rows(&[[0.0, 1.0, 1.0], [0.5, 2.0 / 255.0, 1.0]]);
    let h = histogram(&img);
    assert_eq!(h.len(), 256, "256 grey levels, got {}", h.len());
    assert_eq!(h[0], 1, "level 0: got {}, expected 1", h[0]);
    assert_eq!(h[255], 3, "level 255: got {}, expected 3", h[255]);
    assert_eq!(h[2], 1, "level 2: got {}, expected 1", h[2]);
    assert_eq!(h[128], 1, "0.5 is level round(127.5) = 128: got {}, expected 1", h[128]);
}

#[test]
fn c2_entropy_b2() {
    // p = (0.4, 0.2, 0.2, 0.1, 0.1) as counts 8, 4, 4, 2, 2
    let h = entropy(&[8, 4, 4, 2, 2]);
    assert!(close(h, 2.122, 1e-3), "B2: got {h}, expected 2.122 bits");
}

#[test]
fn c2_entropy_b6_b7() {
    let h = entropy(&[4, 4, 4, 4]);
    assert!(close(h, 2.0, 1e-5), "B6, four levels of 4 pixels: got {h}, expected 2");
    let h = entropy(&[95, 5]);
    assert!(close(h, 0.286, 1e-3), "B7, 95 % background: got {h}, expected 0.286");
    let h = entropy(&[0, 7, 0]);
    assert!(close(h, 0.0, 1e-6), "a constant image: got {h}, expected 0");
}

#[test]
fn c3_huffman_b3() {
    let counts = [8, 4, 4, 2, 2];
    let l = huffman_lengths(&counts);
    assert_eq!(l, vec![2, 2, 2, 3, 3], "B3 (ties: created first): lengths of A..E");
    let total: usize = counts.iter().sum();
    let mean: f32 = counts.iter().zip(&l).map(|(&c, &n)| (c * n) as f32).sum::<f32>() / total as f32;
    assert!(close(mean, 2.2, 1e-5), "B3: average length {mean}, expected 2.2");
}

#[test]
fn c3_huffman_b12_dyadic() {
    let l = huffman_lengths(&[8, 4, 2, 1, 1]);
    assert_eq!(l, vec![1, 2, 3, 4, 4], "B12: lengths for p = (1/2, 1/4, 1/8, 1/16, 1/16)");
}

#[test]
fn c3_huffman_kraft() {
    // Every Huffman tree is full: the sum of 2^(-L) over the symbols is exactly 1.
    let counts = [5, 9, 12, 13, 16, 45, 1, 0, 3];
    let l = huffman_lengths(&counts);
    assert_eq!(l[7], 0, "a symbol with count 0 gets no code word");
    let kraft: f32 = l.iter().filter(|&&n| n > 0).map(|&n| 0.5f32.powi(n as i32)).sum();
    assert!(close(kraft, 1.0, 1e-6), "sum of 2^-L: got {kraft}, expected 1");
    assert_eq!(huffman_lengths(&[0, 3]), vec![0, 1], "one symbol: length 1");
}

#[test]
fn c4_run_lengths_b5() {
    let row: Vec<u8> = "0000000111100000".bytes().map(|b| b - b'0').collect();
    assert_eq!(run_lengths(&row), vec![7, 4, 5], "B5: runs of 0000000111100000");
    let alt: Vec<u8> = "0101010101010101".bytes().map(|b| b - b'0').collect();
    assert_eq!(run_lengths(&alt), vec![1; 16], "B5: the alternating row has 16 runs of 1");
    assert_eq!(run_lengths(&[1, 1, 0]), vec![0, 2, 1], "a row starting with 1 starts with a run of 0 zeros");
}

#[test]
fn c5_arithmetic_b8() {
    // A = 0, B = 1, C = 2 with p = (0.6, 0.3, 0.1); message AAB
    let (lo, hi) = arithmetic_interval(&[0.6, 0.3, 0.1], &[0, 0, 1]);
    assert!((lo - 0.216).abs() < 1e-9 && (hi - 0.324).abs() < 1e-9, "B8: got [{lo}, {hi}), expected [0.216, 0.324)");
    let bits = -(hi - lo).log2();
    assert!((bits - 3.21).abs() < 0.01, "B8: about {bits} bits, expected 3.21");
}
