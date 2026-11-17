// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p matching

use imgcourse::samples;
use matching::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_three_scores_b1() {
    let a = [0., 1., 3., 1., 0., 2., 6., 2.];
    let b = [1., 3., 1.];
    let cc = [6., 11., 6., 3., 12., 22.];
    let ssd = [9., 0., 9., 10., 27., 11.];
    let ncc = [-0.189, 1., -0.189, -0.866, -0.189, 1.];
    for t in 0..6 {
        let s = scores_1d(&a, &b, t as isize);
        assert_eq!(s.n, 3, "t = {t}: the overlap has {} pairs, expected 3", s.n);
        assert!(close(s.cc, cc[t], 1e-4), "B1, t = {t}: CC {}, expected {}", s.cc, cc[t]);
        assert!(close(s.ssd, ssd[t], 1e-4), "B1, t = {t}: SSD {}, expected {}", s.ssd, ssd[t]);
        assert!(close(s.ncc, ncc[t], 1e-3), "B1, t = {t}: NCC {:.3}, expected {}", s.ncc, ncc[t]);
    }
}

#[test]
fn c1_bright_region_and_flat_overlap_b2() {
    let a = [0., 1., 0., 0., 5., 5., 5.];
    let b = [0., 1., 0.];
    let cc = [1., 0., 0., 5., 5.];
    for t in 0..5 {
        let s = scores_1d(&a, &b, t as isize);
        assert!(close(s.cc, cc[t], 1e-4), "B2, t = {t}: CC {}, expected {}", s.cc, cc[t]);
    }
    let s = scores_1d(&a, &b, 4);
    assert!(s.ncc == 0.0, "B2(b): the pixels of A under B are flat, NCC must be 0, got {}", s.ncc);
}

#[test]
fn c1_partial_overlap() {
    // B hangs over both ends of A: only the pairs inside count.
    let a = [1., 2., 3.];
    let b = [5., 1., 2., 3., 7.];
    let s = scores_1d(&a, &b, -1);
    assert_eq!(s.n, 3, "t = −1: overlap of {} pairs, expected 3", s.n);
    assert!(close(s.ssd, 0.0, 1e-6) && close(s.ncc, 1.0, 1e-5), "t = −1: SSD {} and NCC {}, expected 0 and 1", s.ssd, s.ncc);
    let s = scores_1d(&a, &b, -4);
    assert_eq!(s.n, 1, "t = −4: overlap of {} pairs, expected 1 (B's last sample on A's first)", s.n);
}

#[test]
fn c1_same_pattern_b3() {
    let s = scores_1d(&[1., 2., 3.], &[3., 5., 7.], 0);
    assert!(close(s.ssd, 29.0, 1e-4), "B3: SSD {}, expected 29", s.ssd);
    assert!(close(s.ncc, 1.0, 1e-5), "B3: NCC {}, expected 1", s.ncc);
}

#[test]
fn c2_ncc_ignores_gain_and_offset() {
    let cell = samples::cell();
    let a = cell.crop(40, 50, 64, 64);
    let b = cell.crop(45, 47, 64, 64).map(|v| 1.6 * v + 0.2); // B(x) = A(x + (5, −3)), brighter
    let c = ncc_2d(&a, &b, 5, -3, 100).expect("the overlap is large");
    assert!(close(c, 1.0, 1e-4), "at the true shift (5, −3): NCC {c}, expected 1");
    let c = ncc_2d(&a, &b, 0, 0, 100).unwrap();
    assert!(c < 0.9, "at shift (0, 0): NCC {c}, expected clearly below 1");
    assert!(ncc_2d(&a, &b, 60, 60, 100).is_none(), "shift (60, 60) overlaps 16 pixels: expected None with min_overlap 100");
}

#[test]
fn c3_register_finds_the_shift() {
    let cell = samples::cell();
    let a = cell.crop(100, 100, 48, 48);
    let b = cell.crop(107, 96, 48, 48);
    let (tx, ty, c) = register(&a, &b, 10, 200);
    assert!((tx, ty) == (7, -4), "got the shift ({tx}, {ty}), expected (7, −4)");
    assert!(close(c, 1.0, 1e-4), "NCC at the shift: {c}, expected 1");
}

#[test]
fn c4_template_finds_itself() {
    let img = samples::cell().crop(60, 30, 60, 50);
    let tpl = img.crop(21, 13, 12, 10);
    let map = template_match(&img, &tpl);
    assert_eq!((map.width(), map.height()), (49, 41), "the map must have one value per position where the template fits");
    let (mut bx, mut by) = (0, 0);
    for (x, y) in map.coords() {
        if map[(x, y)] > map[(bx, by)] {
            (bx, by) = (x, y);
        }
    }
    assert!((bx, by) == (21, 13), "best match at ({bx}, {by}), expected (21, 13), where the template was cut out");
    assert!(close(map[(21, 13)], 1.0, 1e-4), "NCC at the template's own place: {}, expected 1", map[(21, 13)]);
}
