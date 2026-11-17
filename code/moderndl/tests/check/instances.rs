// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p moderndl instances::

use imgcourse::Image;
use moderndl::instances::*;

/// The 6 × 8 three-class map of B1 ('.' background, 'B' boundary, 'I' interior).
fn b1_map() -> Image<u8> {
    let text = [". B B B B B B .", "B I I B I I I B", "B I I B I I I B", "B I I I I I I B", "B I I B I I I B", ". B B B B B B ."];
    let cells: Vec<Vec<u8>> = text
        .iter()
        .map(|r| {
            r.split(' ')
                .map(|c| match c {
                    "I" => INTERIOR,
                    "B" => BOUNDARY,
                    _ => BACKGROUND,
                })
                .collect()
        })
        .collect();
    Image::from_fn(8, 6, |x, y| cells[y][x])
}

fn count(labels: &Image<u32>) -> usize {
    let mut seen: Vec<u32> = labels.pixels().iter().cloned().filter(|&l| l != 0).collect();
    seen.sort();
    seen.dedup();
    seen.len()
}

#[test]
fn c1_boundary_route_b1() {
    let map = b1_map();
    let labels = boundary_route(&map);
    assert_eq!(count(&labels), 1, "B1 (a): the gap at row 4, column 4 joins the two interiors: one object");
    assert!(labels.coords().all(|p| (labels[p] == 0) == (map[p] == BACKGROUND)), "every interior and boundary pixel should get a label, the background none");
    let mut fixed = map.clone();
    fixed[(3, 3)] = BOUNDARY; // row 4, column 4 counted from 1
    let labels = boundary_route(&fixed);
    assert_eq!(count(&labels), 2, "B1 (b): with the gap closed there should be two objects");
    assert!(labels[(1, 1)] != labels[(5, 1)], "B1 (b): the left and right cells should get different labels");
    assert!(labels[(0, 2)] == labels[(1, 2)] && labels[(7, 2)] == labels[(6, 2)], "the boundary pixels should be grown into their cells");
}

fn profile(values: &[f32]) -> Image<f32> {
    Image::from_fn(values.len(), 1, |x, _| values[x])
}

#[test]
fn c2_distance_route_b2() {
    let two = profile(&[0.0, 2.0, 4.0, 6.0, 5.0, 4.0, 5.0, 3.0, 1.0, 0.0]);
    let labels = distance_route(&two, 0.0, 0.5);
    assert_eq!(count(&labels), 2, "B2 (b): h = 0.5 is below the second peak's dynamic (1): two objects, got {labels}");
    assert!(labels[(3, 0)] != labels[(6, 0)], "B2 (b): the peaks at positions 4 and 7 should be different objects");
    let labels = distance_route(&two, 0.0, 1.5);
    assert_eq!(count(&labels), 1, "B2 (b): h = 1.5 removes the second peak: one object, got {labels}");
    let rod = profile(&[0.0, 3.0, 5.0, 5.2, 4.9, 5.1, 5.0, 3.0, 0.0]);
    assert_eq!(count(&distance_route(&rod, 0.0, 0.5)), 1, "B2 (c): the elongated cell with h = 0.5: one object");
    assert_eq!(count(&distance_route(&rod, 0.0, 0.1)), 2, "B2 (c): the elongated cell with h = 0.1: split in two");
    assert_eq!(distance_route(&two, 0.0, 0.5)[(0, 0)], 0, "pixels at or below the mask threshold are background");
}

#[test]
fn c3_match_objects() {
    // Truth: a 4 × 4 cell and a 2 × 4 cell side by side; prediction: one 6 × 4 blob (a merge).
    let truth = Image::from_fn(8, 4, |x, _| if x < 4 { 1 } else if x < 6 { 2 } else { 0 });
    let merged = Image::from_fn(8, 4, |x, _| if x < 6 { 7 } else { 0 });
    let (tp, fp, fn_, ious) = match_objects(&truth, &merged);
    assert_eq!((tp, fp, fn_), (1, 0, 1), "a merge: the blob still matches the big cell (IoU 16/24), the small one is missed");
    assert!((ious[0] - 2.0 / 3.0).abs() < 1e-5, "IoU of the match: got {}, expected 2/3", ious[0]);
    let (tp, fp, fn_, ious) = match_objects(&truth, &truth);
    assert_eq!((tp, fp, fn_), (2, 0, 0), "the truth matches itself");
    assert!(ious.iter().all(|&v| (v - 1.0).abs() < 1e-6), "with IoU 1, got {ious:?}");
    // Two equal cells merged: IoU exactly 0.5 each, not above it: no match.
    let truth = Image::from_fn(8, 4, |x, _| if x < 4 { 1 } else { 2 });
    let (tp, fp, fn_, _) = match_objects(&truth, &Image::new(8, 4, 3));
    assert_eq!((tp, fp, fn_), (0, 1, 2), "two equal cells merged: IoU 0.5 is not a match");
}

#[test]
fn c4_panoptic_quality_b4() {
    let (sq, rq, pq) = panoptic_quality(&[0.9, 0.8, 0.7], 1, 2);
    assert!((sq - 0.8).abs() < 1e-4 && (rq - 2.0 / 3.0).abs() < 1e-4 && (pq - 0.5333).abs() < 1e-3, "B4: got SQ {sq}, RQ {rq}, PQ {pq}, expected 0.8, 0.667, 0.533");
    let (_, rq, pq) = panoptic_quality(&[0.9, 0.8, 0.7], 0, 2);
    assert!((rq - 0.75).abs() < 1e-4 && (pq - 0.6).abs() < 1e-4, "B4 without the false positive: got RQ {rq}, PQ {pq}, expected 0.75, 0.6");
}
