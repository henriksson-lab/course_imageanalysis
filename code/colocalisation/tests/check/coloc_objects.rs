// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p colocalisation coloc_objects::

use colocalisation::coloc_objects::sim::{self, PIXEL};
use colocalisation::coloc_objects::*;
use imgcourse::{Image, Rng};

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

/// The spots of B1, in pixels.
const A1: [(f32, f32); 5] = [(2., 3.), (5., 8.), (7., 8.), (12., 4.), (15., 15.)];
const B1: [(f32, f32); 6] = [(3., 3.), (6., 9.), (9., 12.), (12., 7.), (18., 15.), (1., 14.)];

#[test]
fn c1_nearest_b1() {
    let nn = nearest(&A1, &B1);
    let expected = [(0, 1.0), (1, 2f32.sqrt()), (1, 2f32.sqrt()), (3, 3.0), (4, 3.0)];
    for (i, (&(j, d), &(ej, ed))) in nn.iter().zip(expected.iter()).enumerate() {
        assert!(j == ej && close(d, ed, 1e-4), "B1(a), A{}: got B{} at {d:.2}, expected B{} at {ed:.2}", i + 1, j + 1, ej + 1);
    }
    let f = fraction_within(&A1, &B1, 2.0);
    assert!(close(f, 0.6, 1e-6), "B1(b): got {f}, expected 0.6");
}

#[test]
fn c2_one_to_one_b1() {
    let n = one_to_one(&A1, &B1, 2.0);
    assert_eq!(n, 2, "B1(c): B2 can serve only one of A2 and A3, so 2 pairs, got {n}");
    let n = one_to_one(&A1, &B1, 3.0);
    assert_eq!(n, 4, "within 3 px: A1–B1, A2 or A3 with B2, A4–B4, A5–B5, got {n}");
}

#[test]
fn c3_chance_level_b2() {
    let sparse = chance_level(0.002, 2.0);
    let dense = chance_level(0.02, 2.0);
    assert!(close(sparse, 0.0248, 1e-3), "B2(a): got {sparse}, expected 0.025");
    assert!(close(dense, 0.222, 1e-3), "B2(a): got {dense}, expected 0.222");
    let (cs, cd) = (corrected(0.30, sparse), corrected(0.30, dense));
    assert!(close(cs, 0.28, 0.005) && close(cd, 0.10, 0.005), "B2(b): got {cs} and {cd}, expected 0.28 and 0.10");
}

#[test]
fn c4_overlap_shares_b3() {
    let a = Image::from_rows(&[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 2, 2, 2, 0, 3, 0],
        [0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
        [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
    ]);
    let b = Image::from_rows(&[
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    .map(|v| v == 1);
    let (any, majority, centroid) = overlap_shares(&a, &b);
    assert!(close(any, 0.75, 1e-6), "B3 (i) any overlap: got {any}, expected 0.75");
    assert!(close(majority, 0.25, 1e-6), "B3 (ii) more than 50 %: got {majority}, expected 0.25");
    assert!(close(centroid, 0.5, 1e-6), "B3 (iii) centroid in B: got {centroid}, expected 0.5");
}

#[test]
fn c5_cross_k_by_hand() {
    // two A spots and two B spots in an area of 100
    let a = [(0.0, 0.0), (10.0, 0.0)];
    let b = [(1.0, 0.0), (10.0, 1.0)];
    // distances: A1–B1 1, A1–B2 √101, A2–B1 9, A2–B2 1
    let k = cross_k(&a, &b, 100.0, 2.0);
    assert!(close(k, 50.0, 1e-4), "2 of 4 pairs within r = 2: K = 100/(2·2)·2 = 50, got {k}");
    let k = cross_k(&a, &b, 100.0, 9.5);
    assert!(close(k, 75.0, 1e-4), "3 of 4 pairs within r = 9.5: K = 75, got {k}");
}

#[test]
fn c5_independent_spots_give_pi_r_squared() {
    // B8: uniform B spots, independent of A: K(r) ≈ π r² (away from the edges)
    let mask = Image::new(64, 64, true);
    let mut rng = Rng::new(3);
    let a = relocate(300, &mask, &mut rng);
    let b = relocate(300, &mask, &mut rng);
    let r = 200.0;
    let l = (cross_k(&a, &b, sim::area(&mask), r) / std::f32::consts::PI).sqrt();
    // without edge correction L is a little low; 64 px × 65 nm = 4160 nm wide, so not by much
    assert!(l > 0.85 * r && l < 1.1 * r, "L(200 nm) of independent spots: got {l:.0} nm, expected about 200 nm");
}

#[test]
fn c5_relocate_stays_inside() {
    let mask = sim::cells(64);
    let mut rng = Rng::new(9);
    let p = relocate(500, &mask, &mut rng);
    assert_eq!(p.len(), 500, "relocate should return n points");
    for &q in &p {
        assert!(sim::in_mask(&mask, q), "the point ({:.0}, {:.0}) nm is outside the mask", q.0, q.1);
    }
    let (w, h) = (64.0 * PIXEL, 64.0 * PIXEL);
    let mx = p.iter().map(|q| q.0).sum::<f32>() / 500.0;
    assert!(mx > 0.2 * w && mx < 0.8 * w && p.iter().all(|q| q.1 < h), "the points should spread over both cells");
}
