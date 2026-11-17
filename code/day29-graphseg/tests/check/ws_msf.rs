// Checks for the coding exercises, using the numbers of the pen-and-paper exercises and the
// demo questions. Pixels a–g of the chain are nodes 0–6; marker A is label 1, B is 2, C is 3.
// Run: cargo test -p day29 ws_msf::

use imgcourse::Image;
use day29::ws_msf::*;

/// B1: the chain a–g with weights 2, 5, 1, 3, 6, 2.
fn b1() -> Vec<Edge> {
    [2.0, 5.0, 1.0, 3.0, 6.0, 2.0].iter().enumerate().map(|(i, &d)| (i, i + 1, d)).collect()
}

/// B3: the 4 × 4 landscape.
fn b3() -> Image<f32> {
    Image::from_rows(&[[1.0, 3.0, 9.0, 4.0], [2.0, 5.0, 8.0, 6.0], [3.0, 7.0, 2.0, 7.0], [4.0, 9.0, 6.0, 0.0]])
}

fn letters(labels: &[u32], width: usize) -> String {
    labels.chunks(width).map(|row| row.iter().map(|&l| b".ABC"[l as usize] as char).collect::<String>()).collect::<Vec<_>>().join(" / ")
}

#[test]
fn c1_altitudes() {
    let e = altitude_edges(&b3());
    assert_eq!(e.len(), 24, "a 4 × 4 grid has 24 edges, got {}", e.len());
    assert_eq!(e[0], (0, 1, 3.0), "first edge: (0, 0)–(1, 0) at max(1, 3) = 3, got {:?}", e[0]);
    assert_eq!(e[1], (0, 4, 2.0), "second edge: (0, 0)–(0, 1) at max(1, 2) = 2, got {:?}", e[1]);
    assert!(e.contains(&(10, 14, 6.0)), "the 2 in row 3 and the 6 below it: altitude 6");
}

#[test]
fn c2_prim_chain_b1() {
    let mut markers = vec![0; 7];
    markers[0] = 1;
    markers[6] = 2;
    let (labels, cuts) = prim_from_markers(7, &b1(), &markers);
    assert_eq!(letters(&labels, 7), "AAAAABB", "B1(b): a–e belong to A, f and g to B");
    assert_eq!(cuts, vec![4], "B1(a): the only cut edge is e–f (weight 6), got the edges {cuts:?}");
}

#[test]
fn c2_prim_landscape_b3_p1() {
    let f = b3();
    let edges = altitude_edges(&f);
    let mut markers = vec![0; 16];
    markers[0] = 1;
    markers[15] = 2;
    let (labels, cuts) = prim_from_markers(16, &edges, &markers);
    assert_eq!(letters(&labels, 4), "AAAB / AAAB / AABB / AABB", "B3(b): the labels");
    let mut alt: Vec<f32> = cuts.iter().map(|&e| edges[e].2).collect();
    alt.sort_by(f32::total_cmp);
    assert_eq!(alt, vec![7.0, 8.0, 8.0, 9.0, 9.0], "B3(c), P1: the altitudes of the cut edges");
}

#[test]
fn c2_prim_third_marker_p2() {
    let edges = altitude_edges(&b3());
    let mut markers = vec![0; 16];
    markers[0] = 1;
    markers[15] = 2;
    markers[10] = 3; // the 2 in row 3, column 3
    let (labels, _) = prim_from_markers(16, &edges, &markers);
    assert_eq!(letters(&labels, 4), "AAAC / AACC / ACCC / AACB", "P2: a third marker on the 2");
}

#[test]
fn c2_prim_plateau_p4() {
    let f = Image::from_rows(&[[1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0]]);
    let mut markers = vec![0; 7];
    markers[0] = 1;
    markers[6] = 2;
    let (labels, _) = prim_from_markers(7, &altitude_edges(&f), &markers);
    assert_eq!(letters(&labels, 7), "AAAABBB", "P4: on a plateau the queue order decides (A was queued first)");
}

#[test]
fn c3_kruskal_equals_prim() {
    let edges = altitude_edges(&b3());
    let mut markers = vec![0; 16];
    markers[0] = 1;
    markers[15] = 2;
    assert_eq!(letters(&kruskal_from_markers(16, &edges, &markers), 4), "AAAB / AAAB / AABB / AABB", "B3 with a virtual root");
    let mut markers = vec![0; 7];
    markers[0] = 1;
    markers[6] = 2;
    assert_eq!(letters(&kruskal_from_markers(7, &b1(), &markers), 7), "AAAAABB", "B1 with a virtual root");
}

#[test]
fn c4_minimax_b1() {
    let from_a = minimax_distance(7, &b1(), 0);
    let from_g = minimax_distance(7, &b1(), 6);
    assert_eq!(from_a[3], 5.0, "B1(c): minimax distance d → A, got {}", from_a[3]);
    assert_eq!(from_g[3], 6.0, "B1(c): minimax distance d → B, got {}", from_g[3]);
    assert_eq!(from_a, vec![0.0, 2.0, 5.0, 5.0, 5.0, 6.0, 6.0], "minimax distances from a");
}

#[test]
fn c4_minimax_decides_the_label_b3() {
    // every pixel goes to a marker it reaches over the lowest highest pass (B3 has many ties
    // at 7: there Prim's queue order decides)
    let edges = altitude_edges(&b3());
    let (da, db) = (minimax_distance(16, &edges, 0), minimax_distance(16, &edges, 15));
    let mut markers = vec![0; 16];
    markers[0] = 1;
    markers[15] = 2;
    let (labels, _) = prim_from_markers(16, &edges, &markers);
    for p in 0..16 {
        let own = if labels[p] == 1 { da[p] } else { db[p] };
        assert_eq!(own, da[p].min(db[p]), "pixel {p}: minimax distances {} to A and {} to B, but it went to {}", da[p], db[p], labels[p]);
    }
    assert_eq!(db[10], 6.0, "B3(a): the 2 reaches B over passes of at most 6, got {}", db[10]);
    assert_eq!(da[10], 7.0, "B3(a): from A the 2 needs a pass of 7, got {}", da[10]);
}
