// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p bvh

use bvh::*;
use imgcourse::Rng;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn unit_box() -> Aabb {
    Aabb { min: [0.0, 0.0], max: [2.0, 1.0] }
}

#[test]
fn c1_slab_test_b1() {
    let (n, f) = slab_test([-1.0, 0.5], [1.0, 0.25], &unit_box()).unwrap_or((f32::NAN, f32::NAN));
    assert!(close(n, 1.0) && close(f, 2.0), "B1 (a): got [{n}, {f}], expected [1, 2]");
    assert!(slab_test([-1.0, 2.0], [1.0, 0.25], &unit_box()).is_none(), "B1 (b): the ray passes above the box");
    let (n, f) = slab_test([1.0, -3.0], [0.0, 1.0], &unit_box()).unwrap_or((f32::NAN, f32::NAN));
    assert!(close(n, 3.0) && close(f, 4.0), "B1 (c), d_x = 0: got [{n}, {f}], expected [3, 4]");
    assert!(slab_test([3.0, -3.0], [0.0, 1.0], &unit_box()).is_none(), "d_x = 0 with the origin outside the x-slab: a miss");
    assert!(slab_test([3.0, 0.5], [1.0, 0.0], &unit_box()).is_none(), "a box behind the ray is a miss");
    // Origin exactly on a slab plane with d_x = 0: 0 · ∞ = NaN must not spoil the test.
    let hit = slab_test([0.0, -3.0], [0.0, 1.0], &unit_box());
    assert!(hit.is_some(), "origin on the plane x = 0, going up along it: should count as a hit, got {hit:?}");
}

/// The eight intervals of B2 as discs on the x axis: a ray along the axis meets each one
/// where its interval starts.
fn b2() -> Vec<Disc> {
    let iv = [[0.0, 1.0], [2.0, 3.0], [3.5, 4.0], [6.0, 7.0], [10.0, 11.0], [12.0, 12.5], [14.0, 15.0], [17.0, 18.0]];
    iv.iter().map(|[a, b]| Disc { centre: [(a + b) / 2.0, 0.0], radius: (b - a) / 2.0 }).collect()
}

fn x_range(b: &Aabb) -> [f32; 2] {
    [b.min[0], b.max[0]]
}

#[test]
fn c2_median_tree_b2() {
    let t = build_median(&b2(), 2);
    let root = &t.nodes[0];
    assert!(!root.is_leaf(), "8 intervals with at most 2 per leaf: the root must be split");
    assert_eq!(x_range(&root.bbox), [0.0, 18.0], "B2 (a): the root");
    let (l, r) = (&t.nodes[root.left], &t.nodes[root.right]);
    assert_eq!(x_range(&l.bbox), [0.0, 7.0], "B2 (a): the left child");
    assert_eq!(x_range(&r.bbox), [10.0, 18.0], "B2 (a): the right child");
    let leaves: Vec<[f32; 2]> = [l.left, l.right, r.left, r.right].iter().map(|&i| x_range(&t.nodes[i].bbox)).collect();
    assert_eq!(leaves, vec![[0.0, 3.0], [3.5, 7.0], [10.0, 12.5], [14.0, 18.0]], "B2 (a): the four leaves");
    assert_eq!(t.nodes.len(), 7, "B2 (a): 7 nodes, got {}", t.nodes.len());
}

#[test]
fn c3_traverse_b2() {
    let discs = b2();
    let t = build_median(&discs, 2);
    let (hit, s) = traverse(&t, &discs, [8.0, 0.0], [1.0, 0.0]);
    let (th, k) = hit.expect("B2 (b): the ray from x = 8 must hit [10, 11]");
    assert!(close(th, 2.0) && k == 4, "B2 (b): got t = {th} on interval {k}, expected t = 2 on interval 4 ([10, 11])");
    assert_eq!((s.box_tests, s.object_tests), (5, 2), "B2 (b): box and interval tests");
    let (hit, s) = traverse(&t, &discs, [-1.0, 0.0], [1.0, 0.0]);
    assert_eq!(hit.map(|h| h.1), Some(0), "B2 (c): the ray from x = -1 must hit [0, 1]");
    assert_eq!((s.box_tests, s.object_tests), (5, 2), "B2 (c): box and interval tests");
    let (hit, s) = traverse(&t, &discs, [0.0, 5.0], [1.0, 0.0]);
    assert!(hit.is_none() && s.box_tests == 1 && s.object_tests == 0, "a ray that misses the root costs one box test, got {s:?}");
}

/// The nearest hit by testing every disc.
fn brute(discs: &[Disc], o: P2, d: P2) -> Option<(f32, usize)> {
    let mut best: Option<(f32, usize)> = None;
    for (k, disc) in discs.iter().enumerate() {
        if let Some(t) = disc.hit(o, d) {
            if best.is_none_or(|b| t < b.0) {
                best = Some((t, k));
            }
        }
    }
    best
}

/// Every disc in exactly one leaf, and every box holds its children's boxes and discs.
fn check_tree(t: &Bvh, discs: &[Disc]) {
    let mut seen = vec![0; discs.len()];
    for n in &t.nodes {
        if n.is_leaf() {
            for &k in &n.objects {
                seen[k] += 1;
                assert!(n.bbox.contains(discs[k].bbox()), "a leaf's box must hold its discs");
            }
        } else {
            assert!(n.bbox.contains(t.nodes[n.left].bbox) && n.bbox.contains(t.nodes[n.right].bbox), "a box must hold its children's boxes");
        }
    }
    assert!(seen.iter().all(|&s| s == 1), "every disc must be in exactly one leaf");
}

fn same_hits_as_brute_force(t: &Bvh, discs: &[Disc]) {
    let mut rng = Rng::new(3);
    for _ in 0..200 {
        let o = [100.0 * rng.uniform(), 100.0 * rng.uniform()];
        let a = std::f32::consts::TAU * rng.uniform();
        let d = [a.cos(), a.sin()];
        let got = traverse(t, discs, o, d).0.map(|h| h.1);
        let expected = brute(discs, o, d).map(|h| h.1);
        assert_eq!(got, expected, "ray from {o:?} along {d:?}: the tree finds {got:?}, testing every disc {expected:?}");
    }
}

#[test]
fn c3_traverse_equals_brute_force() {
    let discs = random_scene(Layout::Uniform, 300, 1);
    same_hits_as_brute_force(&build_median(&discs, 2), &discs);
}

#[test]
fn c4_sah_b3() {
    let a = sah_cost(10.0, 4.0, 6, 5.0, 6);
    let b = sah_cost(10.0, 7.0, 10, 2.0, 2);
    assert!(close(a, 6.4), "B3, split A: got {a}, expected 6.4");
    assert!(close(b, 8.4), "B3, split B: got {b}, expected 8.4");
}

#[test]
fn c4_sah_tree() {
    let discs = random_scene(Layout::Clustered, 256, 2);
    let sah = build_sah(&discs);
    check_tree(&sah, &discs);
    same_hits_as_brute_force(&sah, &discs);
    let median = build_median(&discs, 2);
    assert!(sah.sah_cost() < median.sah_cost(), "the SAH tree should be cheaper by its own measure: SAH {} against median {}", sah.sah_cost(), median.sah_cost());
}

#[test]
fn c5_grid_walk_b4() {
    let cells = grid_walk([0.0, 0.5], [1.0, 0.4], 4, 1.0);
    let expected = [(0, 0, 0.0), (1, 0, 1.0), (1, 1, 1.25), (2, 1, 2.0), (3, 1, 3.0), (3, 2, 3.75)];
    assert_eq!(cells.len(), expected.len(), "B4: got the cells {cells:?}, expected {expected:?}");
    for (g, e) in cells.iter().zip(expected) {
        assert!(g.0 == e.0 && g.1 == e.1 && close(g.2, e.2), "B4: got {g:?}, expected {e:?}");
    }
    let down = grid_walk([2.5, 3.5], [0.0, -1.0], 4, 1.0);
    let rows: Vec<usize> = down.iter().map(|c| c.1).collect();
    assert_eq!(rows, vec![3, 2, 1, 0], "straight down from row 3: rows {rows:?}");
}
