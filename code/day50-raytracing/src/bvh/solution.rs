// Model solution for the coding exercises of "Tracing fast: boxes, trees and grids".

use crate::bvh::scene2d::{Aabb, Bvh, Disc, Node, Stats, P2};

/// C1. The slab test: the interval `(t_near, t_far)` of the ray `o + t d` inside the box,
/// or `None` if the ray misses it (`t_near > t_far`, or the box lies behind, `t_far ≤ 0`).
/// `t_near` may be negative when the origin is inside the box. A direction component of 0
/// needs no special case: `1/0 = ∞` in floating point.
pub fn slab_test(o: P2, d: P2, b: &Aabb) -> Option<(f32, f32)> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    for k in 0..2 {
        let inv = 1.0 / d[k];
        let mut t0 = (b.min[k] - o[k]) * inv;
        let mut t1 = (b.max[k] - o[k]) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        // Written so that a NaN (0 · ∞) never replaces the current value.
        if t0 > t_near {
            t_near = t0;
        }
        if t1 < t_far {
            t_far = t1;
        }
    }
    if t_near <= t_far && t_far > 0.0 {
        Some((t_near, t_far))
    } else {
        None
    }
}

/// C2. Build a BVH by median splits: the box of a node holds all its discs; if it has more
/// than `max_leaf` discs, sort them by their centres along the longer side of the box of the
/// centres, and give the first half to the left child, the rest to the right child.
pub fn build_median(discs: &[Disc], max_leaf: usize) -> Bvh {
    fn build(discs: &[Disc], mut ids: Vec<usize>, max_leaf: usize, nodes: &mut Vec<Node>) -> usize {
        let bbox = ids.iter().fold(Aabb::empty(), |b, &i| b.union(discs[i].bbox()));
        let me = nodes.len();
        nodes.push(Node { bbox, left: 0, right: 0, objects: vec![] });
        if ids.len() <= max_leaf {
            nodes[me].objects = ids;
            return me;
        }
        // The longer side of the box around the centres.
        let centres = ids.iter().fold(Aabb::empty(), |b, &i| b.union(Aabb { min: discs[i].centre, max: discs[i].centre }));
        let axis = if centres.max[0] - centres.min[0] >= centres.max[1] - centres.min[1] { 0 } else { 1 };
        ids.sort_by(|&a, &b| discs[a].centre[axis].total_cmp(&discs[b].centre[axis]));
        let right_ids = ids.split_off(ids.len() / 2);
        let left = build(discs, ids, max_leaf, nodes);
        let right = build(discs, right_ids, max_leaf, nodes);
        nodes[me].left = left;
        nodes[me].right = right;
        me
    }
    let mut nodes = Vec::new();
    build(discs, (0..discs.len()).collect(), max_leaf, &mut nodes);
    Bvh { nodes }
}

/// C3. The nearest hit of the ray `o + t d` (`t`, index of the disc), by traversing the
/// tree: test the root's box; at an inner node test both children's boxes and visit the
/// nearer child first; skip a node whose box starts beyond the best hit so far. Counts the
/// box tests and the disc tests.
pub fn traverse(bvh: &Bvh, discs: &[Disc], o: P2, d: P2) -> (Option<(f32, usize)>, Stats) {
    let mut stats = Stats::default();
    let mut best: Option<(f32, usize)> = None;
    let mut best_t = f32::INFINITY;
    stats.box_tests += 1;
    let Some((t_root, _)) = slab_test(o, d, &bvh.nodes[0].bbox) else {
        return (None, stats);
    };
    // A stack of (node, where the ray enters its box); the top is visited next.
    let mut stack = vec![(0, t_root)];
    while let Some((i, t_enter)) = stack.pop() {
        if t_enter > best_t {
            continue; // the box starts beyond the best hit: skip it
        }
        let node = &bvh.nodes[i];
        if node.is_leaf() {
            for &k in &node.objects {
                stats.object_tests += 1;
                if let Some(t) = discs[k].hit(o, d) {
                    if t < best_t {
                        best_t = t;
                        best = Some((t, k));
                    }
                }
            }
            continue;
        }
        stats.box_tests += 2;
        let l = slab_test(o, d, &bvh.nodes[node.left].bbox).map(|(t, _)| (node.left, t));
        let r = slab_test(o, d, &bvh.nodes[node.right].bbox).map(|(t, _)| (node.right, t));
        match (l, r) {
            (Some(a), Some(b)) => {
                // Push the farther one first, so that the nearer one is visited first.
                let (near, far) = if a.1 <= b.1 { (a, b) } else { (b, a) };
                stack.push(far);
                stack.push(near);
            }
            (Some(a), None) | (None, Some(a)) => stack.push(a),
            (None, None) => {}
        }
    }
    (best, stats)
}

/// C4a. The SAH cost of splitting a node of size `parent` (perimeter in 2D, surface area in
/// 3D) into children of sizes `left` and `right` holding `n_left` and `n_right` objects:
/// one box test plus, for each child, (its size ÷ the parent's) × its number of objects.
pub fn sah_cost(parent: f32, left: f32, n_left: usize, right: f32, n_right: usize) -> f32 {
    1.0 + left / parent * n_left as f32 + right / parent * n_right as f32
}

/// C4b. Build a BVH with the surface area heuristic: at every node, try every split of the
/// discs sorted along x and along y (the first `i` against the rest), and take the cheapest
/// by `sah_cost`. If no split is cheaper than a leaf (cost = number of discs), make a leaf.
pub fn build_sah(discs: &[Disc]) -> Bvh {
    fn build(discs: &[Disc], mut ids: Vec<usize>, nodes: &mut Vec<Node>) -> usize {
        let n = ids.len();
        let bbox = ids.iter().fold(Aabb::empty(), |b, &i| b.union(discs[i].bbox()));
        let me = nodes.len();
        nodes.push(Node { bbox, left: 0, right: 0, objects: vec![] });
        // The best split so far: (cost, axis, number of discs on the left). A leaf costs n.
        let mut best = (n as f32, 0, 0);
        for axis in 0..2 {
            ids.sort_by(|&a, &b| discs[a].centre[axis].total_cmp(&discs[b].centre[axis]));
            // Boxes of the first i discs, and of the discs from i on.
            let mut left_box = vec![Aabb::empty(); n + 1];
            let mut right_box = vec![Aabb::empty(); n + 1];
            for i in 0..n {
                left_box[i + 1] = left_box[i].union(discs[ids[i]].bbox());
                right_box[n - 1 - i] = right_box[n - i].union(discs[ids[n - 1 - i]].bbox());
            }
            for i in 1..n {
                let cost = sah_cost(bbox.perimeter(), left_box[i].perimeter(), i, right_box[i].perimeter(), n - i);
                if cost < best.0 {
                    best = (cost, axis, i);
                }
            }
        }
        let (_, axis, split) = best;
        if split == 0 {
            nodes[me].objects = ids; // a leaf is cheapest
            return me;
        }
        ids.sort_by(|&a, &b| discs[a].centre[axis].total_cmp(&discs[b].centre[axis]));
        let right_ids = ids.split_off(split);
        let left = build(discs, ids, nodes);
        let right = build(discs, right_ids, nodes);
        nodes[me].left = left;
        nodes[me].right = right;
        me
    }
    let mut nodes = Vec::new();
    build(discs, (0..discs.len()).collect(), &mut nodes);
    Bvh { nodes }
}

/// C5. Walk the ray through a grid of `cells` × `cells` square cells of side `cell_size`
/// covering `[0, cells · cell_size]²`, starting from the cell that holds `o` (inside the
/// grid). Returns the cells `(column, row)` in the order the ray visits them, each with the
/// `t` at which the ray enters it (0 for the first).
pub fn grid_walk(o: P2, d: P2, cells: usize, cell_size: f32) -> Vec<(usize, usize, f32)> {
    let mut cell = [0isize; 2];
    let mut step = [0isize; 2];
    let mut t_next = [f32::INFINITY; 2]; // t at which the ray crosses the next boundary
    let mut t_delta = [f32::INFINITY; 2]; // t between two boundaries
    for k in 0..2 {
        cell[k] = ((o[k] / cell_size).floor() as isize).clamp(0, cells as isize - 1);
        if d[k] > 0.0 {
            step[k] = 1;
            t_next[k] = ((cell[k] + 1) as f32 * cell_size - o[k]) / d[k];
            t_delta[k] = cell_size / d[k];
        } else if d[k] < 0.0 {
            step[k] = -1;
            t_next[k] = (cell[k] as f32 * cell_size - o[k]) / d[k];
            t_delta[k] = -cell_size / d[k];
        }
    }
    let mut visited = vec![(cell[0] as usize, cell[1] as usize, 0.0)];
    loop {
        // Cross the boundary that comes first.
        let k = if t_next[0] <= t_next[1] { 0 } else { 1 };
        if t_next[k] == f32::INFINITY {
            break;
        }
        let t = t_next[k];
        cell[k] += step[k];
        if cell[k] < 0 || cell[k] >= cells as isize {
            break; // left the grid
        }
        t_next[k] += t_delta[k];
        visited.push((cell[0] as usize, cell[1] as usize, t));
    }
    visited
}
