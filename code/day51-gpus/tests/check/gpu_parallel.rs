// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day51 gpu_parallel::

use day51::gpu_parallel::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_warp_cycles_b2() {
    let lanes: Vec<bool> = (0..32).map(|k| (8..=12).contains(&k)).collect();
    assert_eq!(warp_cycles(&lanes, 100, 10), 110, "B2 (a): 5 lanes inside, 27 outside: both paths");
    assert_eq!(warp_cycles(&[true; 32], 100, 10), 100, "all lanes inside: only path A");
    assert_eq!(warp_cycles(&[false; 32], 100, 10), 10, "all lanes outside: only path B");
}

#[test]
fn c2_one_row_b2() {
    let mask = Image::from_fn(32, 1, |x, _| (8..=12).contains(&x));
    let d = divergence(&mask, 32, 1, 100, 10);
    assert_eq!((d.warps, d.diverged, d.cycles), (1, 1, 110), "B2 (a): one warp, diverged, 110 cycles; got {d:?}");
    assert!(close(d.efficiency, 0.21875, 1e-4), "B2 (b): efficiency 770/3520 = 21.9 %, got {}", d.efficiency);
}

#[test]
fn c2_tiles_b2c() {
    // A 32 × 4 image; the object covers exactly the tile x = 8..16.
    let mask = Image::from_fn(32, 4, |x, _| (8..16).contains(&x));
    let tiles = divergence(&mask, 8, 4, 100, 10);
    assert_eq!((tiles.warps, tiles.diverged, tiles.cycles), (4, 0, 130), "B2 (c), 8 × 4 tiles: 4 warps, none diverged, 100 + 3 · 10 cycles; got {tiles:?}");
    assert!(close(tiles.efficiency, 1.0, 1e-6), "B2 (c): efficiency 100 %, got {}", tiles.efficiency);
    let rows = divergence(&mask, 32, 1, 100, 10);
    assert_eq!((rows.warps, rows.diverged), (4, 4), "the same object with warps = rows: every row diverges; got {rows:?}");
}

#[test]
fn c2_stripes_b5() {
    let vertical = Image::from_fn(64, 8, |x, _| (x / 8) % 2 == 0);
    let horizontal = Image::from_fn(64, 8, |_, y| y % 2 == 0);
    let shapes = [(32, 1, "rows of 32"), (8, 4, "8 × 4 tiles")];
    let expected = [[16, 0], [0, 16]]; // diverged warps: [vertical, horizontal] per shape
    for (s, &(w, h, name)) in shapes.iter().enumerate() {
        let v = divergence(&vertical, w, h, 100, 10).diverged;
        let hz = divergence(&horizontal, w, h, 100, 10).diverged;
        assert_eq!([v, hz], expected[s], "B5, {name}: diverged warps for vertical and horizontal stripes");
    }
}

#[test]
fn c3_slowest_lane_b4() {
    let (warp, eff) = loop_efficiency(&[3, 1, 1, 12, 2, 1, 4, 1]);
    assert_eq!(warp, 12, "B4: the warp runs as long as its slowest lane");
    assert!(close(eff, 25.0 / 96.0, 1e-4), "B4: efficiency 25/96 = 26 %, got {eff}");
}

#[test]
fn c4_latency_hiding() {
    let one = utilisation(1, 4, 200, 6);
    assert!(close(one, 4.0 / 204.0, 1e-3), "one warp, k = 4, L = 200: busy 4 of every 204 cycles, got {one}");
    let half = utilisation(13, 8, 200, 6);
    assert!(close(half, 13.0 * 8.0 / 208.0, 0.06), "13 warps, k = 8, L = 200: about N k / (k + L) = 0.5, got {half}");
    let full = utilisation(32, 8, 200, 6);
    assert!(full > 0.8, "32 warps, k = 8, L = 200: enough to hide the latency (start and end aside), got {full}");
}
