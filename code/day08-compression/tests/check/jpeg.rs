// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day08 jpeg::

use imgcourse::Image;
use day08::jpeg::*;

/// The block of B1: c00 = 190, c01 = −33, c10 = 25, c20 = −15, c11 = 5, c02 = 9 (row, column),
/// all others small. In the code, (u, v) = (column, row).
fn block_b1() -> Image<f32> {
    let mut c = Image::new(8, 8, 0.0);
    c[(0, 0)] = 190.0;
    c[(1, 0)] = -33.0;
    c[(0, 1)] = 25.0;
    c[(0, 2)] = -15.0;
    c[(1, 1)] = 5.0;
    c[(2, 0)] = 9.0;
    c[(5, 5)] = 3.0; // "smaller than half its step"
    c
}

#[test]
fn c1_scale_table_b3() {
    for (q, q00, q77) in [(10, 80, 255), (25, 32, 198), (50, 16, 99), (75, 8, 50), (100, 1, 1)] {
        let t = scale_table(q);
        assert_eq!((t[0][0], t[7][7]), (q00, q77), "B3, quality {q}: Q00 and Q77 were {} and {}, expected {q00} and {q77}", t[0][0], t[7][7]);
    }
}

#[test]
fn c2_quantise_b1() {
    let table = scale_table(50);
    let q = quantise(&block_b1(), &table);
    let expected = [((0, 0), 12), ((1, 0), -3), ((0, 1), 2), ((0, 2), -1), ((1, 1), 0), ((2, 0), 1)];
    for ((u, v), e) in expected {
        assert_eq!(q[(u, v)], e, "B1(a), row {v}, column {u}: got {}, expected {e}", q[(u, v)]);
    }
    let nonzero = q.pixels().iter().filter(|&&v| v != 0).count();
    assert_eq!(nonzero, 5, "B1(a): {nonzero} non-zero values, expected 5");
    let back = dequantise(&q, &table);
    let expected = [((0, 0), 192.0), ((1, 0), -33.0), ((0, 1), 24.0), ((0, 2), -14.0), ((1, 1), 0.0), ((2, 0), 10.0)];
    for ((u, v), e) in expected {
        assert_eq!(back[(u, v)], e, "B1(b), row {v}, column {u}: decoded {}, expected {e}", back[(u, v)]);
    }
}

#[test]
fn c3_zigzag_b2() {
    let q = quantise(&block_b1(), &scale_table(50));
    let zz = zigzag(&q);
    assert_eq!(zz.len(), 64, "64 values");
    assert_eq!(&zz[..7], &[12, -3, 2, -1, 0, 1, 0], "B2(a): the start of the zigzag sequence");
    // The order itself: number every position by row * 8 + column and read it in zigzag.
    let pos = Image::from_fn(8, 8, |x, y| (y * 8 + x) as i32);
    let order = zigzag(&pos);
    assert_eq!(&order[..10], &[0, 1, 8, 16, 9, 2, 3, 10, 17, 24], "zigzag order of the first 10 positions (row · 8 + column)");
    assert_eq!(order[63], 63, "the last position is (7, 7)");
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, (0..64).collect::<Vec<i32>>(), "every position exactly once");
}

#[test]
fn c4_size_of() {
    for (v, s) in [(0, 0), (1, 1), (-1, 1), (2, 2), (3, 2), (-3, 2), (4, 3), (7, 3), (8, 4), (-1023, 10)] {
        assert_eq!(size_of(v), s, "size of {v}: got {}, expected {s}", size_of(v));
    }
}

#[test]
fn c4_ac_symbols_b2() {
    let mut zz = vec![0; 64];
    zz[..6].copy_from_slice(&[12, -3, 2, -1, 0, 1]);
    let s = ac_symbols(&zz);
    assert_eq!(s, vec![(0, 2, -3), (0, 2, 2), (0, 1, -1), (1, 1, 1), EOB], "B2(b): the AC symbols");
    // a long run: 20 zeros before a 5, then nothing
    let mut zz = vec![0; 64];
    zz[21] = 5;
    assert_eq!(ac_symbols(&zz), vec![ZRL, (4, 3, 5), EOB], "20 zeros = ZRL (16) + run 4");
    let mut zz = vec![0; 64];
    zz[63] = -2;
    assert_eq!(ac_symbols(&zz), vec![ZRL, ZRL, ZRL, (14, 2, -2)], "a non-zero last value: no EOB");
}

#[test]
fn c5_psnr_b5() {
    let a = Image::new(3, 3, 100.0);
    let errors = Image::from_rows(&[[2.0, -1.0, 0.0], [0.0, 3.0, 0.0], [-2.0, 1.0, 0.0]]);
    let b = a.zip_map(&errors, |p, e| p + e);
    let p = psnr(&a, &b, 255.0);
    assert!((p - 44.9).abs() < 0.05, "B5: got {p:.2} dB, expected 44.9");
}
