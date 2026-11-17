// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Nodes A–F are 0–5; the pixels of a 1D signal are numbered from 0 (the exercises from 1).
// Run: cargo test -p mst-seg

use imgcourse::Image;
use mst_seg::*;

/// B1: AB 3, AD 1, BC 5.5, BD 2.5, BE 2, CF 4, DE 5, EF 7.
fn b1() -> Vec<Edge> {
    let (a, b, c, d, e, f) = (0, 1, 2, 3, 4, 5);
    vec![(a, b, 3.0), (a, d, 1.0), (b, c, 5.5), (b, d, 2.5), (b, e, 2.0), (c, f, 4.0), (d, e, 5.0), (e, f, 7.0)]
}

/// A chain with d = |difference| between neighbours, left to right.
fn chain(values: &[f32]) -> Vec<Edge> {
    (0..values.len() - 1).map(|i| (i, i + 1, (values[i + 1] - values[i]).abs())).collect()
}

/// The regions as a string of letters a, b, c, … in order of first appearance.
fn pattern(labels: &[usize]) -> String {
    let mut seen: Vec<usize> = Vec::new();
    labels
        .iter()
        .map(|l| {
            let k = seen.iter().position(|s| s == l).unwrap_or_else(|| {
                seen.push(*l);
                seen.len() - 1
            });
            (b'a' + k as u8) as char
        })
        .collect()
}

const B2: [f32; 12] = [0.0, 6.0, 0.0, 6.0, 20.0, 20.0, 20.0, 20.0, 24.0, 24.0, 24.0, 24.0];
const B3: [f32; 8] = [10.0, 11.0, 13.0, 12.0, 22.0, 23.0, 21.0, 24.0];

#[test]
fn c1_union_find() {
    let mut uf = UnionFind::new(6);
    assert!(uf.union(0, 3).is_some() && uf.union(1, 4).is_some() && uf.union(3, 4).is_some(), "joining different components should succeed");
    assert!(uf.union(0, 1).is_none(), "0 and 1 are already in one component: union should return None");
    assert_eq!(uf.find(0), uf.find(4), "0 and 4 should have the same root");
    assert_ne!(uf.find(0), uf.find(2), "0 and 2 should have different roots");
    let r = uf.find(1);
    assert_eq!(uf.size[r], 4, "the component {{0, 1, 3, 4}} has 4 nodes, got size {}", uf.size[r]);
}

#[test]
fn c2_kruskal_b1() {
    let tree = kruskal(6, &b1());
    let names = ["A", "B", "C", "D", "E", "F"];
    let joined: Vec<String> = tree.iter().map(|&(a, b, _)| format!("{}{}", names[a], names[b])).collect();
    assert_eq!(joined, ["AD", "BE", "BD", "CF", "BC"], "B1(a): joined edges in order");
    let total: f32 = tree.iter().map(|e| e.2).sum();
    assert!((total - 15.0).abs() < 1e-5, "B1(b): got total weight {total}, expected 15");
}

#[test]
fn c3_single_linkage_b1() {
    assert_eq!(pattern(&single_linkage(6, &b1(), 3.0)), "aabaac", "B1(c), t = 3: {{A, B, D, E}}, {{C}}, {{F}}");
    assert_eq!(pattern(&single_linkage(6, &b1(), 5.0)), "aabaab", "B1(c), t = 5: {{A, B, D, E}}, {{C, F}}");
}

#[test]
fn c3_single_linkage_ramp_b5() {
    let img = Image::from_fn(8, 4, |x, _| [0.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 15.0][x]);
    let edges = grid_edges(&img, false);
    let count = |l: Vec<usize>| l.iter().max().unwrap() + 1;
    assert_eq!(count(single_linkage(32, &edges, 3.0)), 6, "B5(a), t = 3: expected 6 regions");
    assert_eq!(count(single_linkage(32, &edges, 4.0)), 1, "B5(a), t = 4: expected 1 region (chaining)");
    assert_eq!(count(felzenszwalb(32, &edges, 8.0)), 6, "B5(b), FH with k = 8: expected 6 regions");
}

#[test]
fn c4_fh_b2() {
    assert_eq!(pattern(&felzenszwalb(12, &chain(&B2), 8.0)), "aaaabbbbcccc", "B2(c), k = 8: texture, 20s, 24s");
    assert_eq!(pattern(&felzenszwalb(12, &chain(&B2), 15.9)), "aaaabbbbcccc", "B2(d), k = 15.9: still three regions");
    assert_eq!(pattern(&felzenszwalb(12, &chain(&B2), 16.0)), "aaaabbbbbbbb", "B2(d), k = 16: the 20s and 24s merge");
    assert_eq!(pattern(&felzenszwalb(12, &chain(&B2), 4.0)), "abcdeeeeffff", "G3, k = 4: the texture breaks up");
}

#[test]
fn c4_fh_b3_b4() {
    assert_eq!(pattern(&felzenszwalb(8, &chain(&B3), 4.0)), "aaaabbbb", "B3(a), k = 4");
    assert_eq!(pattern(&felzenszwalb(8, &chain(&B3), 31.9)), "aaaabbbb", "B3(c), k = 31.9: still two regions");
    assert_eq!(pattern(&felzenszwalb(8, &chain(&B3), 32.0)), "aaaaaaaa", "B3(c), k = 32: one region");
    let doubled: Vec<f32> = B3.iter().map(|v| 2.0 * v).collect();
    assert_eq!(pattern(&felzenszwalb(8, &chain(&doubled), 4.0)), "aaaabbbc", "B4(a), doubled values, k = 4");
}

#[test]
fn c5_merge_small() {
    // FH with k = 4 leaves the texture pixels of B2 alone; a minimum size of 3 joins them up
    let edges = chain(&B2);
    let fh = felzenszwalb(12, &edges, 4.0);
    let cleaned = merge_small(12, &edges, &fh, 3);
    let p = pattern(&cleaned);
    assert_eq!(p, "aaaabbbbcccc", "the texture pixels should join up, the two flat parts (4 pixels each) stay apart");
    let mut sizes = std::collections::HashMap::new();
    for l in &cleaned {
        *sizes.entry(*l).or_insert(0) += 1;
    }
    assert!(sizes.values().all(|&s| s >= 3), "every region should have at least 3 pixels, got {p}");
}
