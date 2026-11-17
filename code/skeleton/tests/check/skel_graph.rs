// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p skeleton skel_graph::
//
// The exercise writes pixels as (row, column); the code indexes img[(x, y)] = img[(column, row)].

use imgcourse::Image;
use skeleton::skel_graph::*;

/// A binary image from rows of '0' and '1'.
fn grid(rows: &[&str]) -> Image<bool> {
    Image::from_fn(rows[0].len(), rows.len(), |x, y| rows[y].as_bytes()[x] == b'1')
}

/// The skeleton of B1.
fn b1() -> Image<bool> {
    grid(&["000000000", "010000011", "001000100", "000101010", "000010000", "000010000", "000010000", "000010000", "000000000"])
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The weighted lengths of all edges, sorted.
fn lengths(g: &Graph) -> Vec<f32> {
    let mut v: Vec<f32> = g.edges.iter().map(|e| path_steps(&e.path).weighted()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

#[test]
fn c1_neighbour_count_b1() {
    let n = neighbour_count(&b1());
    // (row, column) -> expected count
    let expected = [((1, 1), 1), ((2, 2), 2), ((3, 3), 2), ((4, 4), 3), ((5, 4), 2), ((7, 4), 1), ((3, 5), 2), ((2, 6), 3), ((1, 7), 2), ((1, 8), 1), ((3, 7), 1)];
    for ((r, c), e) in expected {
        assert_eq!(n[(c, r)], e, "B1(a) pixel (row, column) = ({r}, {c})");
    }
    assert_eq!(n[(0, 0)], 0, "off the skeleton the count is 0");
}

#[test]
fn c2_path_steps_b3() {
    let diagonal: Vec<(usize, usize)> = (0..=50).map(|i| (i, i)).collect();
    let s = path_steps(&diagonal);
    assert_eq!(s, Steps { ne: 0, no: 50, nc: 0 }, "B3(a) the diagonal line");
    assert!(close(s.pixel_steps(), 50.0, 1e-3) && close(s.weighted(), 70.71, 0.01) && close(s.corrected(), 70.30, 0.01), "B3(a) estimates: got {}, {}, {}", s.pixel_steps(), s.weighted(), s.corrected());

    // slope 1/2: axial, diagonal, axial, diagonal, ... (100 steps)
    let mut path = vec![(0usize, 0usize)];
    for i in 0..100 {
        let (x, y) = *path.last().unwrap();
        path.push(if i % 2 == 0 { (x + 1, y) } else { (x + 1, y + 1) });
    }
    let s = path_steps(&path);
    assert_eq!(s, Steps { ne: 50, no: 50, nc: 99 }, "B3(b) the line of slope 1/2");
    assert!(close(s.weighted(), 120.71, 0.01) && close(s.corrected(), 110.29, 0.01), "B3(b) estimates: got {}, {}", s.weighted(), s.corrected());

    let horizontal: Vec<(usize, usize)> = (0..=100).map(|x| (x, 7)).collect();
    let s = path_steps(&horizontal);
    assert!(s == Steps { ne: 100, no: 0, nc: 0 } && close(s.corrected(), 98.0, 1e-3), "B3(c) the horizontal line: got {s:?}");
}

#[test]
fn c3_graph_b1() {
    let g = build_graph(&b1());
    assert_eq!((g.nodes.len(), g.edges.len()), (6, 5), "B1(b) 6 nodes (4 end points, 2 junctions) and 5 edges");
    let got = lengths(&g);
    let expected = [1.414, 2.414, 2.828, 3.0, 4.243];
    for (a, b) in got.iter().zip(expected) {
        assert!(close(*a, b, 0.01), "B1(c) weighted edge lengths: got {got:?}, expected {expected:?}");
    }
    let steps: f32 = g.edges.iter().map(|e| path_steps(&e.path).pixel_steps()).sum();
    assert!(close(steps, 11.0, 1e-3), "B1(c) total length in pixel steps: got {steps}, expected 11");
    for e in &g.edges {
        let (first, last) = (e.path[0], *e.path.last().unwrap());
        assert!(g.nodes[e.a].pixels.contains(&first) && g.nodes[e.b].pixels.contains(&last), "an edge path must run from a pixel of node a to a pixel of node b: {e:?}");
    }
}

#[test]
fn c3_junction_cluster_is_one_node() {
    // A cross whose middle is thick: the pixels there all have 3 or more neighbours.
    let thick = grid(&["0001000", "0001000", "0011100", "1111111", "0011100", "0001000", "0001000"]);
    let g = build_graph(&thick);
    let junctions: Vec<&Node> = g.nodes.iter().filter(|n| n.pixels.len() > 1).collect();
    assert_eq!(junctions.len(), 1, "the touching junction pixels in the middle form one node: got {:?}", g.nodes);
    assert_eq!(g.edges.len(), 4, "four arms");
}

/// A diamond-shaped loop |dx| + |dy| = r around (cx, cy).
fn diamond(img: &mut Image<bool>, cx: usize, cy: usize, r: usize) {
    for (x, y) in img.coords().collect::<Vec<_>>() {
        if x.abs_diff(cx) + y.abs_diff(cy) == r {
            img[(x, y)] = true;
        }
    }
}

#[test]
fn c4_cycles_b1_b4() {
    assert_eq!(cycles(&build_graph(&b1())), 0, "B1(d) a tree has no cycles");

    // B4(a): a θ, a loop with a bar across it.
    let mut theta = Image::new(20, 20, false);
    diamond(&mut theta, 10, 10, 7);
    for x in 4..=16 {
        theta[(x, 10)] = true;
    }
    let g = build_graph(&theta);
    assert_eq!(cycles(&g), 2, "B4(a) the θ has E − V + C = 2: got E = {}, V = {}", g.edges.len(), g.nodes.len());

    // B4(b): a ring (no nodes) and a separate Y.
    let mut two = Image::new(30, 20, false);
    diamond(&mut two, 7, 9, 5);
    let (cx, cy) = (22, 8);
    for i in 1..=4 {
        two[(cx - i, cy - i)] = true;
        two[(cx + i, cy - i)] = true;
        two[(cx, cy + i)] = true;
    }
    two[(cx, cy)] = true;
    let g = build_graph(&two);
    assert_eq!((g.edges.len(), g.nodes.len()), (4, 5), "B4(b) ring (1 node, 1 loop edge) and Y (4 nodes, 3 edges)");
    assert_eq!(cycles(&g), 1, "B4(b) E − V + C = 4 − 5 + 2 = 1");
}

#[test]
fn c5_prune_b2() {
    for k in [2.0, 3.0] {
        let pruned = prune(&b1(), k);
        let mut expected = b1();
        expected[(7, 3)] = false; // (row, column) = (3, 7)
        assert_eq!(pruned, expected, "B2 k = {k}: only the spur to (3, 7) goes");
        let g = build_graph(&pruned);
        assert_eq!((g.nodes.len(), g.edges.len()), (4, 3), "B2(a) after pruning the graph is a Y");
        assert!(lengths(&g).iter().any(|&l| close(l, 5.243, 0.01)), "B2(b) the merged edge (1, 8) to (4, 4) is 5.24 long: got {:?}", lengths(&g));
    }
    assert_eq!(prune(&b1(), 1.0), b1(), "k = 1 removes nothing");
}
