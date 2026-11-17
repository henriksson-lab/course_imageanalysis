// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p cellnets sam::

use imgcourse::Image;
use cellnets::sam::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_upsample_keeps_constants_and_ramps() {
    let flat = Image::new(4, 4, 2.5f32);
    let up = upsample_logits(&flat, 16, 16);
    assert!(up.pixels().iter().all(|&v| close(v, 2.5, 1e-5)), "a constant map should stay 2.5 everywhere");
    // a ramp along x: coarse column i has the value i; 4 → 16 pixels, factor 4
    let ramp = Image::from_fn(4, 4, |x, _| x as f32);
    let up = upsample_logits(&ramp, 16, 16);
    for x in 2..14 {
        let want = (x as f32 + 0.5) / 4.0 - 0.5;
        assert!(close(up[(x, 5)], want, 1e-5), "ramp, pixel ({x}, 5): got {}, expected {want}", up[(x, 5)]);
    }
    assert!(close(up[(0, 0)], 0.0, 1e-5) && close(up[(15, 0)], 3.0, 1e-5), "ramp at the edges: got {} and {}, expected 0 and 3 (clamp)", up[(0, 0)], up[(15, 0)]);
}

/// A logit map with `a` pixels at +2, `b` at 0 and the rest at −2.
fn logits(a: usize, b: usize) -> Image<f32> {
    Image::from_fn(50, 20, |x, y| {
        let i = y * 50 + x;
        if i < a { 2.0 } else if i < a + b { 0.0 } else { -2.0 }
    })
}

#[test]
fn c2_stability_b2() {
    let p = stability(&logits(380, 50), 1.0);
    let q = stability(&logits(400, 12), 1.0);
    assert!(close(p, 380.0 / 430.0, 1e-5), "mask P: got {p}, expected 380/430 = 0.88 (B2a)");
    assert!(close(q, 400.0 / 412.0, 1e-5), "mask Q: got {q}, expected 400/412 = 0.97 (B2a)");
    let none = stability(&Image::new(5, 5, -3.0), 1.0);
    assert!(close(none, 0.0, 1e-6), "no pixel above −1: got {none}, expected 0");
}

#[test]
fn c3_click_selects_one_blob() {
    // two blobs of positive logits, separated by a negative column
    let l = Image::from_fn(9, 5, |x, y| if x == 4 || y == 0 { -1.0 } else { 1.0 });
    let m = mask_from_click(&l, 1, 2);
    let n = m.pixels().iter().filter(|&&v| v).count();
    assert_eq!(n, 16, "click on the left blob: got {n} pixels, expected the 4 × 4 left blob");
    assert!(!m[(6, 2)], "the right blob should not be in the mask of a click on the left blob");
    let empty = mask_from_click(&l, 4, 2);
    assert!(empty.pixels().iter().all(|&v| !v), "a click on a negative logit should give an empty mask");
}

fn cand(bbox: (usize, usize, usize, usize), score: f32, stability: f32) -> Candidate {
    Candidate { pixels: Vec::new(), bbox, score, stability }
}

#[test]
fn c4_filters() {
    let cands = vec![
        cand((10, 10, 29, 29), 0.95, 0.97), // 0: good
        cand((11, 10, 30, 29), 0.90, 0.98), // 1: duplicate of 0 (box IoU 0.90)
        cand((40, 10, 59, 29), 0.85, 0.90), // 2: unstable
        cand((70, 10, 89, 29), 0.75, 0.99), // 3: low score
        cand((15, 15, 34, 34), 0.88, 0.96), // 4: overlaps 0 by box IoU 0.39: kept
    ];
    let kept = filter_masks(&cands, 0.8, 0.95, 0.7);
    assert_eq!(kept, vec![0, 4], "kept {kept:?}, expected [0, 4]: 1 is a duplicate, 2 unstable, 3 below the score");
}
