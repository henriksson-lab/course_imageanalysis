// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day03 highbit::

use day03::highbit::*;

#[test]
fn c1_wrap_and_saturate_b2() {
    let cases = [((1200, 1500), 65236, 0), ((90, 100), 65526, 0), ((100, 90), 10, 10), ((150, 160), 65526, 0)];
    for ((a, b), wrap, sat) in cases {
        assert_eq!(sub_wrap(a, b), wrap, "B2: {a} − {b} with wrap-around: got {}, expected {wrap}", sub_wrap(a, b));
        assert_eq!(sub_saturate(a, b), sat, "B2: {a} − {b} with saturation: got {}, expected {sat}", sub_saturate(a, b));
    }
    let mean = add_wrap(40000, 30000) / 2;
    assert_eq!(mean, 2232, "B2(d): (40 000 + 30 000) / 2 with wrap-around: got {mean}, expected 2232");
    assert_eq!(add_wrap(65535, 1), 0, "65 535 + 1 wraps to 0");
    // B6: adding 65 506 = 0xFFE2 is subtracting 30.
    for x in [0u16, 29, 30, 1000, 65535] {
        assert_eq!(add_wrap(x, 0xFFE2), sub_wrap(x, 30), "B6: {x} + 65 506 should equal {x} − 30 (mod 2^16)");
    }
    assert_eq!(0xFFE2u16 as i16, -30, "B6: 0xFFE2 read as int16 is −30");
}

#[test]
fn c2_pack12_b3() {
    let b = pack12(0xABC, 0x123);
    assert_eq!(b, [0xAB, 0xC1, 0x23], "B3: packing 0xABC and 0x123: got {b:02X?}, expected [AB, C1, 23]");
    for (p1, p2) in [(0xABC, 0x123), (0, 4095), (4095, 0), (1234, 2345)] {
        assert_eq!(unpack12(pack12(p1, p2)), (p1, p2), "unpack12 should undo pack12 for ({p1}, {p2})");
    }
}

#[test]
fn c3_window_b1_b4() {
    // B1
    assert_eq!(window(4095.0, 0.0, 65535.0), 16, "B1: 4095 with the window 0 … 65 535");
    assert_eq!(window(4095.0, 0.0, 4095.0), 255, "B1: 4095 with the window 0 … 4095");
    assert_eq!(window(65520.0, 0.0, 65535.0), 255, "B1: 65 520 (MSB-aligned 4095) with the window 0 … 65 535");
    // B4: the soft-tissue window W = 400, L = 40, i.e. −160 … 240.
    for (hu, level) in [(-1000.0, 0), (-100.0, 38), (0.0, 102), (40.0, 128), (700.0, 255)] {
        let got = window(hu, -160.0, 240.0);
        assert_eq!(got, level, "B4, soft-tissue window: {hu} HU gives level {got}, expected {level}");
    }
    // B4: the lung window W = 1500, L = −600, i.e. −1350 … 150.
    for (hu, level) in [(-850.0, 85), (40.0, 236)] {
        let got = window(hu, -1350.0, 150.0);
        assert_eq!(got, level, "B4, lung window: {hu} HU gives level {got}, expected {level}");
    }
}

#[test]
fn c4_percentile() {
    let values: Vec<u16> = (0..=100).rev().collect(); // 100, 99, …, 0
    assert_eq!(percentile(&values, 0.0), 0, "the 0th percentile is the smallest value");
    assert_eq!(percentile(&values, 1.0), 100, "the 100th percentile is the largest value");
    assert_eq!(percentile(&values, 0.5), 50, "the median of 0 … 100");
    assert_eq!(percentile(&values, 0.995), 100, "99.5 % of 0 … 100 (position 99.5 rounds to 100)");
    assert_eq!(percentile(&[7, 3, 5], 0.5), 5, "the median of 7, 3, 5");
}
