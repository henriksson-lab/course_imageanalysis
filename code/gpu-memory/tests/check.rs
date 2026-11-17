// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p gpu-memory

use gpu_memory::*;
use imgcourse::{samples, Image};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// The byte address of float pixel (x, y) in a row-major image `width` pixels wide.
fn addr(x: usize, y: usize, width: usize) -> usize {
    4 * (y * width + x)
}

#[test]
fn p1_coalescing_b3() {
    let column: Vec<usize> = (0..32).map(|y| addr(5, y, 1024)).collect();
    let t = warp_traffic(&column);
    assert_eq!((t.lines, t.sectors, t.bytes_moved, t.bytes_used), (32, 32, 1024, 128), "B3 (a), a column: got {t:?}");
    assert!(close(t.efficiency as f64, 0.125, 1e-6), "B3 (a): efficiency 12.5 %, got {}", t.efficiency);
    let row: Vec<usize> = (0..32).map(|x| addr(x, 5, 1024)).collect();
    let t = warp_traffic(&row);
    assert_eq!((t.lines, t.sectors, t.bytes_moved), (1, 4, 128), "B3 (b), a row: got {t:?}");
    let red: Vec<usize> = (0..32).map(|x| 3 * addr(x, 0, 1024)).collect(); // interleaved RGB: 12 bytes per pixel
    let t = warp_traffic(&red);
    assert_eq!((t.lines, t.sectors, t.bytes_moved), (3, 12, 384), "B3 (c), red of interleaved RGB: got {t:?}");
    assert!(close(t.efficiency as f64, 1.0 / 3.0, 1e-6), "B3 (c): efficiency 33 %, got {}", t.efficiency);
}

#[test]
fn p2_tiles_b4() {
    let reads = tile_reads(16, 16, 16, 5);
    let total: u32 = reads.pixels().iter().sum();
    assert_eq!((reads.width(), reads.height()), (20, 20), "B4: the input with its halo is 20 × 20");
    assert_eq!(total, 400, "B4: one 16 × 16 tile with a 5 × 5 filter loads 400 pixels");
    let direct: u32 = tile_reads(16, 16, 1, 5).pixels().iter().sum();
    assert_eq!(direct, 25 * 256, "without tiles (tile = 1) every output reads its own 25 inputs");
    // The lecture's heat map: 32 × 32 outputs, 16 × 16 tiles, 3 × 3 filter.
    let r = tile_reads(32, 32, 16, 3);
    let per_output = r.pixels().iter().sum::<u32>() as f64 / 1024.0;
    assert!(close(per_output, 1.2656, 1e-3), "32 × 32, tiles of 16, 3 × 3: got {per_output} reads per output, expected 1.266");
    assert_eq!((r[(5, 5)], r[(16, 5)], r[(16, 16)]), (1, 2, 4), "inside a tile 1 read, on a seam 2, at the centre corner 4");
}

#[test]
fn p3_roofline_b2() {
    let (peak, bw) = (3e13, 1e12); // the desktop preset
    let n = 4096.0 * 4096.0;
    let (t, mem) = roofline_time(n, 12.0 * n, peak, bw);
    assert!(close(t * 1e6, 201.3, 0.5) && mem, "B2 (a), adding two images: got {:.1} µs, memory-bound {mem}; expected 201 µs, memory-bound", t * 1e6);
    let (t, mem) = roofline_time(18.0 * n, 8.0 * n, peak, bw);
    assert!(close(t * 1e6, 134.2, 0.5) && mem, "B2 (d), 3 × 3 filter with reuse: got {:.1} µs; expected 134 µs, memory-bound", t * 1e6);
    let (t, mem) = roofline_time(2.0 * 121.0 * n, 8.0 * n, peak, bw);
    assert!(!mem && close(t * 1e6, 135.3, 0.5), "an 11 × 11 filter (I = 30.25): compute-bound, got {:.1} µs, memory-bound {mem}", t * 1e6);
}

#[test]
fn p4_fusing_b6() {
    let img = samples::cell();
    let n = img.width() * img.height();
    let (mut ms, mut mf) = (Meter::default(), Meter::default());
    let a = pipeline_separate(&img, 1.5, 0.1, 0.5, &mut ms);
    let b = pipeline_fused(&img, 1.5, 0.1, 0.5, &mut mf);
    assert_eq!(ms.bytes, 32 * n, "B6 (a): four passes move 32 bytes per pixel, got {}", ms.bytes as f64 / n as f64);
    assert_eq!(mf.bytes, 8 * n, "B6 (b): the fused pass moves 8 bytes per pixel, got {}", mf.bytes as f64 / n as f64);
    assert!(a.pixels() == b.pixels(), "fusing must not change the result");
    let expected = Image::from_rows(&[[0.0, 0.2, 0.5]]).map(|v| if (v * 1.5 - 0.1f32).clamp(0.0, 1.0) > 0.5 { 1.0 } else { 0.0 });
    let got = pipeline_fused(&Image::from_rows(&[[0.0, 0.2, 0.5]]), 1.5, 0.1, 0.5, &mut Meter::default());
    assert!(got.pixels() == expected.pixels(), "scale, subtract, clip, threshold of 0, 0.2, 0.5: got {:?}, expected {:?}", got.pixels(), expected.pixels());
}
