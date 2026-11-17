// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p gpus gpu_fit::

use gpus::gpu_fit::*;
use imgcourse::Image;

/// The 6 × 6 mask of B3.
fn b3() -> Image<bool> {
    let rows = [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
    ];
    Image::from_rows(&rows).map(|v: i32| v == 1)
}

#[test]
fn c1a_one_pass_b3a() {
    let mask = b3();
    let start = Image::from_fn(6, 6, |x, y| if mask[(x, y)] { (6 * y + x + 1) as u32 } else { 0 });
    let (after, changed) = propagate_once(&mask, &start);
    assert!(changed, "the first pass must change something");
    let row3: Vec<u32> = (0..6).map(|x| after[(x, 2)]).collect();
    assert_eq!(row3, vec![13, 13, 14, 15, 0, 12], "B3 (a): row 3 after one pass");
}

#[test]
fn c1b_passes_b3b() {
    let (labels, passes) = label_propagation(&b3());
    assert_eq!(passes, 20, "B3 (b): label 1 needs 20 passes to reach the far end of the path");
    for (x, y) in labels.coords() {
        let expected = if b3()[(x, y)] { 1 } else { 0 };
        assert_eq!(labels[(x, y)], expected, "pixel ({x}, {y}): one component, all labels 1");
    }
}

#[test]
fn c1b_bar() {
    // B8 (a): a straight bar of 10 pixels, smallest label at one end: 9 passes.
    let bar = Image::from_fn(10, 1, |_, _| true);
    assert_eq!(label_propagation(&bar).1, 9, "a bar of 10 pixels");
    let two = Image::from_fn(7, 1, |x, _| x != 3);
    let (labels, _) = label_propagation(&two);
    assert_eq!(labels.row(0), &[1, 1, 1, 0, 5, 5, 5], "two separate bars keep two labels");
}

#[test]
fn c2_pointer_jumping() {
    let (labels, passes) = label_pointer_jumping(&b3());
    let (reference, slow) = label_propagation(&b3());
    assert!(labels.pixels() == reference.pixels(), "pointer jumping must give the same labels as plain propagation");
    assert!(passes < slow, "pointer jumping should need fewer passes: got {passes}, plain propagation {slow}");
    assert_eq!(passes, 9, "B3 (c): 9 passes with pointer jumping");
    let bar = Image::from_fn(64, 1, |_, _| true);
    let (_, p) = label_pointer_jumping(&bar);
    assert!(p < 63, "a bar of 64 pixels: plain propagation needs 63 passes, pointer jumping got {p}");
}

#[test]
fn c3_wavefront_b4() {
    let cost = Image::from_fn(7, 5, |x, y| ((3 * x + 5 * y) % 4) as f32);
    let (d, cells) = dp_wavefront(&cost);
    assert_eq!(cells.len(), 11, "B4: 5 + 7 − 1 = 11 steps, got {cells:?}");
    assert_eq!(cells.iter().max(), Some(&5), "B4: at most 5 cells in one step, got {cells:?}");
    assert_eq!(cells.iter().sum::<usize>(), 35, "every cell once");
    // The same table filled row by row.
    let mut r = Image::new(7, 5, 0.0f32);
    for y in 0..5 {
        for x in 0..7 {
            let mut best = f32::INFINITY;
            if x > 0 { best = best.min(r[(x - 1, y)]); }
            if y > 0 { best = best.min(r[(x, y - 1)]); }
            if x > 0 && y > 0 { best = best.min(r[(x - 1, y - 1)]); }
            r[(x, y)] = cost[(x, y)] + if best.is_finite() { best } else { 0.0 };
        }
    }
    for (x, y) in r.coords() {
        assert_eq!(d[(x, y)], r[(x, y)], "cell ({x}, {y}): wavefront {} but row by row {}", d[(x, y)], r[(x, y)]);
    }
}

#[test]
fn c4_amdahl_b2() {
    // Load 2 s, filter 6 s, label 1 s, measure 1 s.
    // Only the filter (60 %) is 50 times faster: serial fraction 0.4.
    let a = amdahl(0.4, 50.0);
    assert!((a - 10.0 / 4.12).abs() < 1e-3, "B2 (a): got a speed-up of {a}, expected 10/4.12 = 2.43");
    let b = amdahl(0.2, 50.0);
    assert!((b - 10.0 / 2.16).abs() < 1e-3, "B2 (b): got {b}, expected 10/2.16 = 4.63");
    let c = amdahl(0.2, 1e12);
    assert!((c - 5.0).abs() < 1e-6, "B2 (c): the limit 1/s = 5, got {c}");
}

#[test]
fn c5_float16_sums_b5() {
    assert_eq!(to_f16(4097.0), 4096.0, "B5 (a): 4097 in float16");
    assert_eq!(to_f16(65535.0), f32::INFINITY, "B5 (a): 65 535 in float16");
    let values = vec![0.01f32; 10000];
    let s16 = sum_f16(&values);
    assert_eq!(s16, 32.0, "B5 (b): the float16 sum stalls at 32, got {s16}");
    let mixed = sum_mixed(&values);
    assert!((mixed - 100.02).abs() < 0.01, "B5 (b): float16 values, float32 sum: got {mixed}, expected 100.02");
}

