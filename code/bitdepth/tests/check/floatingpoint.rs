// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p bitdepth floatingpoint::

use bitdepth::floatingpoint::*;

fn f32_add(v: f32) -> f32 {
    v // float32 additions round by themselves
}

#[test]
fn c1_decode_b1() {
    let cases = [(0x3C00, 1.0), (0xC000, -2.0), (0x7BFF, 65504.0), (0x7C00, f32::INFINITY), (0x0001, 2f32.powi(-24)), (0x2E66, 1638.0 / 16384.0)]; // 0.099 975 585 937 5
    for (bits, value) in cases {
        let got = decode_f16(bits);
        assert_eq!(got, value, "B1: 0x{bits:04X} should be {value}, got {got}");
    }
    assert!(decode_f16(0x7E00).is_nan(), "0x7E00 (E = 31, F ≠ 0) should be NaN");
}

#[test]
fn c2_round_b2_b3() {
    // B2 and B3: (value, nearest float16)
    let cases = [(2049.0, 2048.0), (2050.0, 2050.0), (4095.0, 4096.0), (4097.0, 4096.0), (4100.0, 4100.0), (1000.0, 1000.0), (3001.0, 3000.0), (40001.0, 40000.0)];
    for (x, expected) in cases {
        let got = round_to_f16(x);
        assert_eq!(got, expected, "B2/B3: {x} in float16 should be {expected}, got {got}");
    }
    assert_eq!(round_to_f16(65535.0), f32::INFINITY, "B3: 65 535 overflows to infinity");
    assert_eq!(round_to_f16(-3001.0), -3000.0, "negative numbers round the same way");
    assert_eq!(round_to_f16(0.1), 1638.0 / 16384.0, "E1: 0.1 in float16");
    assert_eq!(round_to_f16(2f32.powi(-25)), 0.0, "E4: half the smallest subnormal is a tie, and rounds to 0");
}

#[test]
fn c2_every_float16_rounds_to_itself() {
    // Every finite float16 value is its own nearest float16.
    for bits in 0..=0xFFFFu16 {
        let v = decode_f16(bits);
        if v.is_finite() {
            let r = round_to_f16(v);
            assert_eq!(r, v, "0x{bits:04X} = {v} should round to itself, got {r}");
        }
    }
}

#[test]
fn c3_naive_sum_stalls_b4() {
    let ones = vec![1.0f32; 10000];
    let s = sum_naive(&ones, round_to_f16);
    assert_eq!(s, 2048.0, "B4: adding 1 ten thousand times in float16 should stop at 2048, got {s}");
    assert_eq!(sum_naive(&ones, f32_add), 10000.0, "B4: in float32 the count is exact, got {}", sum_naive(&ones, f32_add));
    let p = sum_pairwise(&ones, round_to_f16);
    assert_eq!(p, 10000.0, "pairwise in float16 should reach 10 000, got {p}");
}

#[test]
fn c3_kahan_b9() {
    let xs = [2048.0, 1.0, 1.0, 1.0, 1.0];
    let k = sum_kahan(&xs, round_to_f16);
    assert_eq!(k, 2052.0, "B9: Kahan from 2048, adding 1 four times in float16, should give 2052, got {k}");
    assert_eq!(sum_naive(&xs, round_to_f16), 2048.0, "B9: the naive sum stays at 2048");
}

#[test]
fn c3_long_float32_sums() {
    // 2^20 pixels of about 1000: naive float32 drifts, pairwise and Kahan do not.
    let mut rng = imgcourse::Rng::new(3);
    let xs: Vec<f32> = (0..1 << 20).map(|_| (1000.0 + 50.0 * rng.normal()).round()).collect();
    let exact: f64 = xs.iter().map(|&x| x as f64).sum();
    let rel = |s: f32| ((s as f64 - exact) / exact).abs();
    let (n, p, k) = (sum_naive(&xs, f32_add), sum_pairwise(&xs, f32_add), sum_kahan(&xs, f32_add));
    assert!(rel(n) > 5e-6, "naive float32 sum of 2^20 values: relative error {:.1e}, expected a visible drift", rel(n));
    assert!(rel(p) < 1e-6 && rel(k) < 1e-6, "pairwise ({:.1e}) and Kahan ({:.1e}) should be accurate to better than 1e-6", rel(p), rel(k));
}

#[test]
fn c4_variance_b5() {
    let xs = [10000.0f32, 10001.0, 10002.0];
    let one = variance_one_pass(&xs);
    assert!(one <= 0.0, "B5(a): the one-pass variance in float32 should come out 0 or negative, got {one}");
    let two = variance_two_pass(&xs, sum_naive);
    assert!((two - 2.0 / 3.0).abs() < 1e-6, "B5(b): the two-pass variance should be 2/3, got {two}");
}
