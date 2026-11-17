// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p measurements relations::

use imgcourse::Image;
use measurements::relations::*;
use std::collections::BTreeSet;

/// The label image of B1 (x = column, y = row).
fn b1() -> Image<u32> {
    Image::from_rows(&[
        [1, 1, 0, 2, 2, 2],
        [1, 1, 0, 2, 2, 2],
        [1, 1, 3, 0, 0, 0],
        [0, 3, 3, 3, 0, 4],
        [0, 3, 3, 0, 4, 4],
        [5, 0, 0, 0, 4, 4],
    ])
}

fn set(v: &[u32]) -> BTreeSet<u32> {
    v.iter().copied().collect()
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_neighbours_b1() {
    let n4 = neighbours(&b1(), false);
    let expected4 = [set(&[]), set(&[3]), set(&[]), set(&[1]), set(&[]), set(&[])];
    assert_eq!(n4, expected4, "B1(a) 4-connectivity: only 1 and 3 touch");
    let n8 = neighbours(&b1(), true);
    let expected8 = [set(&[]), set(&[3]), set(&[3]), set(&[1, 2, 4, 5]), set(&[3]), set(&[3])];
    assert_eq!(n8, expected8, "B1(a) 8-connectivity: object 3 gains 2, 4 and 5 diagonally");
}

#[test]
fn c2_percent_touching_b1() {
    let p1 = percent_touching(&b1(), 1);
    let p5 = percent_touching(&b1(), 5);
    assert!(close(p1, 50.0, 1e-3), "B1(b) object 1: got {p1} %, expected 50 % (3 of 6 pixels)");
    assert!(close(p5, 100.0, 1e-3), "B1(b) object 5: got {p5} %, expected 100 %");
    let p2 = percent_touching(&b1(), 2);
    assert!(close(p2, 100.0 / 6.0, 1e-3), "object 2: one of its 6 boundary pixels touches 3: got {p2} %");
}

/// The five centroids of B2, as (x, y).
const B2: [[f32; 2]; 5] = [[2., 2.], [5., 2.], [2., 6.], [8., 8.], [9., 3.]];

#[test]
fn c3_closest_two_b2() {
    let c = closest_two(&B2);
    // (first, d1, second, d2), objects numbered from 0 here (object 1 of the exercise is 0)
    let expected = [(1, 3.0, 2, 4.0), (0, 3.0, 4, 4.123), (0, 4.0, 1, 5.0), (4, 5.099, 2, 6.325), (1, 4.123, 3, 5.099)];
    for (i, (got, e)) in c.iter().zip(expected).enumerate() {
        let ok = got.0 == e.0 && got.2 == e.2 && close(got.1, e.1, 1e-3) && close(got.3, e.3, 1e-3);
        assert!(ok, "B2(a) object {}: got {got:?}, expected closest {} at {}, second {} at {} (numbered from 1)", i + 1, e.0 + 1, e.1, e.2 + 1, e.3);
    }
}

#[test]
fn c3_angle_b2() {
    let a = angle_at(B2[0], B2[1], B2[2]);
    assert!(close(a, 90.0, 1e-3), "B2(b) angle at object 1: got {a}°, expected 90°");
    let a = angle_at([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]);
    assert!(close(a, 180.0, 1e-3), "opposite directions: got {a}°, expected 180°");
}

#[test]
fn c3_clark_evans_b2() {
    let r = clark_evans(&B2, 100.0);
    assert!(close(r, 1.72, 0.01), "B2(c) Clark–Evans ratio: got {r}, expected 3.84 / 2.24 = 1.72");
}

#[test]
fn c4_expand_two_far_cells_b3() {
    // Two cells far apart in a large image: expansion makes them neighbours (B3).
    let mut labels = Image::new(40, 20, 0u32);
    labels[(3, 10)] = 1;
    labels[(36, 4)] = 2;
    assert!(neighbours(&labels, true)[1].is_empty(), "not adjacent before expansion");
    let e = expand_labels(&labels);
    assert!(e.pixels().iter().all(|&l| l != 0), "after expansion no background is left");
    assert_eq!(neighbours(&e, false)[1], set(&[2]), "B3: after expansion, 1 and 2 are neighbours");
    assert!(e[(0, 0)] == 1 && e[(39, 19)] == 2 && e[(15, 10)] == 1 && e[(25, 4)] == 2, "pixels go to the nearest object");
}

#[test]
fn c4_expand_keeps_objects() {
    let e = expand_labels(&b1());
    for (x, y) in b1().coords() {
        if b1()[(x, y)] != 0 {
            assert_eq!(e[(x, y)], b1()[(x, y)], "object pixel ({x}, {y}) must keep its label");
        }
    }
    assert_eq!(e[(5, 2)], 2, "pixel (5, 2) is 1 from object 2 and 1 from object 4: the tie goes to the first found, 2");
    assert_eq!(e[(0, 4)], 3, "pixel (0, 4): 1 from objects 3 and 5; 3 comes first in the image");
}

/// The nuclei N and spots S of B4.
fn b4() -> (Image<u32>, Image<u32>) {
    let n = Image::from_rows(&[
        [1, 1, 1, 0, 2, 2, 2],
        [1, 1, 1, 0, 2, 2, 2],
        [1, 1, 1, 0, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0, 0, 0],
        [0, 3, 3, 3, 0, 0, 0],
    ]);
    let s = Image::from_rows(&[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 2, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 3, 3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4],
    ]);
    (n, s)
}

#[test]
fn c5_parent_by_overlap_b4() {
    let (n, s) = b4();
    let p = parent_by_overlap(&s, &n);
    let expected = [(1, 1.0), (1, 0.5), (3, 1.0 / 3.0), (0, 0.0)];
    for c in 1..=4 {
        let (e, g) = (expected[c - 1], p[c]);
        assert!(g.0 == e.0 && close(g.1, e.1, 1e-4), "B4(a) spot {c}: got parent {} with fraction {}, expected {} with {}", g.0, g.1, e.0, e.1);
    }
}

#[test]
fn c5_parent_by_centroid_b4() {
    let (n, s) = b4();
    let p = parent_by_centroid(&s, &n);
    assert_eq!(&p[1..], &[1, 0, 0, 0], "B4(b) parents by centroid of spots 1 to 4");
}
