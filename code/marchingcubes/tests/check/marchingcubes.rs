// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p marchingcubes marchingcubes::

use imgcourse::Volume;
use marchingcubes::marchingcubes::*;
use std::collections::HashMap;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const B1: [f32; 8] = [7.0, 3.0, 2.0, 1.0, 6.0, 4.0, 1.0, 0.0];

fn triangles(values: [f32; 8], c: f32, rule: FaceRule) -> usize {
    cube_loops(values, c, rule).iter().map(|l| l.len() - 2).sum()
}

#[test]
fn r1_case_b1() {
    assert_eq!(cube_case(B1, 5.0), 17, "B1(a): corners 0 and 4 are above 5: expected case 17");
    let mut one = [0.2; 8];
    one[0] = 0.8;
    assert_eq!(cube_case(one, 0.5), 1, "only corner 0 above: case 1");
    assert_eq!(cube_case([0.8; 8], 0.5), 255, "all corners above: case 255");
}

#[test]
fn r2_vertices_b1() {
    let expected = [(0, [0.5, 0.0, 0.0]), (4, [0.0, 0.4, 0.0]), (2, [0.5, 0.0, 1.0]), (6, [0.0, 0.2, 1.0])];
    for (e, p) in expected {
        let got = edge_vertex(B1, 5.0, e);
        assert!(got.is_some_and(|g| (0..3).all(|i| close(g[i], p[i], 1e-5))), "B1(c): edge e{e}: got {got:?}, expected {p:?}");
    }
    for e in [1, 3, 5, 7, 8, 9, 10, 11] {
        assert!(edge_vertex(B1, 5.0, e).is_none(), "B1(b): edge e{e} is not crossed");
    }
}

#[test]
fn r3_one_cube_b1_b4() {
    let loops = cube_loops(B1, 5.0, FaceRule::Separate);
    assert_eq!(loops.len(), 1, "B1(d): one loop expected, got {loops:?}");
    let mut edges = loops[0].clone();
    edges.sort();
    assert_eq!(edges, vec![0, 2, 4, 6], "B1(d): the loop must run through e0, e2, e4, e6; got {:?}", loops[0]);
    assert_eq!(triangles(B1, 5.0, FaceRule::Separate), 2, "B1(d): a quadrilateral, 2 triangles");
    let mut one = [0.2; 8];
    one[0] = 0.8;
    assert_eq!(triangles(one, 0.5, FaceRule::Separate), 1, "B4(a): case 1 gives 1 triangle");
}

#[test]
fn r3_ambiguous_face_c4() {
    // Case 9: corners 0 and 3 above (0.8), the rest 0.2; the face z = 0 is a saddle with f* = 0.5.
    let mut v = [0.2; 8];
    v[0] = 0.8;
    v[3] = 0.8;
    assert_eq!(triangles(v, 0.5, FaceRule::Separate), 2, "C4: case 9 with separate: 2 triangles");
    assert_eq!(triangles(v, 0.5, FaceRule::Join), 4, "C4: case 9 with join: 4 triangles");
    // (C4 uses c = 0.5, exactly f*; c = 0.51 keeps rounding out of the way.)
    assert_eq!(triangles(v, 0.51, FaceRule::Decider), 2, "C4: f* = 0.5 < c = 0.51, so the decider does not join: 2 triangles");
    v[0] = 0.9;
    assert_eq!(triangles(v, 0.51, FaceRule::Decider), 4, "C4: with I0 = 0.9, f* = 0.523 > 0.51: the decider joins, 4 triangles");
}

#[test]
fn r3_every_case_hexagon_c3() {
    // Case 23 (corners 0, 1, 2, 4): a single hexagon.
    let v: [f32; 8] = std::array::from_fn(|k| if [0, 1, 2, 4].contains(&k) { 0.8 } else { 0.2 });
    let loops = cube_loops(v, 0.5, FaceRule::Separate);
    assert!(loops.len() == 1 && loops[0].len() == 6, "C3: case 23 must be one loop of 6 edges, got {loops:?}");
    // In every case, every crossed edge is in exactly one loop.
    for case in 0..256usize {
        let v: [f32; 8] = std::array::from_fn(|k| if case & (1 << k) != 0 { 0.8 } else { 0.2 });
        let mut seen: Vec<usize> = cube_loops(v, 0.5, FaceRule::Separate).concat();
        seen.sort();
        let crossed: Vec<usize> = (0..12).filter(|&e| (v[EDGES[e].0] > 0.5) != (v[EDGES[e].1] > 0.5)).collect();
        assert_eq!(seen, crossed, "case {case}: the loops must pass through each crossed edge exactly once");
    }
}

#[test]
fn r4_gradient_normal_b5() {
    // The six face neighbours of voxel (1, 1, 1) as in B5.
    let mut vol = Volume::new(3, 3, 3, 0.0f32);
    vol[(2, 1, 1)] = 80.0;
    vol[(0, 1, 1)] = 40.0;
    vol[(1, 2, 1)] = 60.0;
    vol[(1, 0, 1)] = 60.0;
    vol[(1, 1, 2)] = 90.0;
    vol[(1, 1, 0)] = 30.0;
    let g = gradient(&vol, [1.0, 1.0, 1.0], 1, 1, 1);
    assert!(close(g[0], 20.0, 1e-4) && close(g[1], 0.0, 1e-4) && close(g[2], 30.0, 1e-4), "B5(a): got ∇I = {g:?}, expected (20, 0, 30)");
    let g = gradient(&vol, [1.0, 1.0, 2.0], 1, 1, 1);
    assert!(close(g[0], 20.0, 1e-4) && close(g[2], 15.0, 1e-4), "B5(b): spacing (1, 1, 2): got ∇I = {g:?}, expected (20, 0, 15)");
    // At the border: a one-sided difference.
    let g = gradient(&vol, [1.0, 1.0, 1.0], 2, 1, 1);
    assert!(close(g[0], 80.0, 1e-4), "at the border x = 2: got ∂I/∂x = {}, expected (I[2] − I[1]) / 1 = 80 − 0 = 80", g[0]);
}

/// Counts (vertices, edges, faces) and checks that every edge has exactly two triangles, which
/// run along it in opposite directions (a closed, consistently oriented surface).
fn check_closed(mesh: &Mesh, what: &str) -> i64 {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for t in &mesh.triangles {
        for i in 0..3 {
            *directed.entry((t[i], t[(i + 1) % 3])).or_default() += 1;
        }
    }
    for (&(a, b), &n) in &directed {
        assert_eq!(n, 1, "{what}: the edge {a} → {b} is used by {n} triangles in the same direction (inconsistent orientation)");
        assert!(directed.contains_key(&(b, a)), "{what}: the edge {a}–{b} has a triangle on one side only (a crack or hole)");
    }
    mesh.vertices.len() as i64 - (directed.len() / 2) as i64 + mesh.triangles.len() as i64
}

#[test]
fn r5_sphere_closed_and_outward() {
    // A ball of radius 4.3 voxels in a 12³ volume (value 1 at the centre, falling off linearly).
    let vol = Volume::from_fn(12, 12, 12, |x, y, z| {
        let d = ((x as f32 - 5.5).powi(2) + (y as f32 - 5.5).powi(2) + (z as f32 - 5.5).powi(2)).sqrt();
        (1.0 - d / 8.6).max(0.0)
    });
    let mesh = marching_cubes(&vol, [1.0, 1.0, 1.0], 0.5, FaceRule::Decider);
    assert!(!mesh.triangles.is_empty(), "the sphere has no triangles");
    let chi = check_closed(&mesh, "sphere");
    assert_eq!(chi, 2, "sphere: V − E + F = {chi}, expected 2");
    let ratio = mesh.triangles.len() as f32 / mesh.vertices.len() as f32;
    assert!((1.9..2.0).contains(&ratio), "a closed sphere has about 2 triangles per vertex (2V − 4): got {ratio:.3}");
    for (p, n) in mesh.vertices.iter().zip(&mesh.normals) {
        let r = ((p[0] - 5.5).powi(2) + (p[1] - 5.5).powi(2) + (p[2] - 5.5).powi(2)).sqrt();
        assert!(close(r, 4.3, 0.15), "every vertex must lie near radius 4.3: got {r} at {p:?}");
        let out = ((p[0] - 5.5) * n[0] + (p[1] - 5.5) * n[1] + (p[2] - 5.5) * n[2]) / r;
        assert!(out > 0.9, "the normal at {p:?} must point outwards: got {n:?}");
    }
}

#[test]
fn r5_torus_and_spacing() {
    // A torus (R = 6, r = 2.5 voxels) in a 20 × 20 × 10 volume: χ = 0.
    let torus = Volume::from_fn(20, 20, 10, |x, y, z| {
        let (dx, dy, dz) = (x as f32 - 9.5, y as f32 - 9.5, z as f32 - 4.5);
        let ring = ((dx * dx + dy * dy).sqrt() - 6.0).hypot(dz);
        (1.0 - ring / 5.0).max(0.0)
    });
    let mesh = marching_cubes(&torus, [1.0, 1.0, 1.0], 0.5, FaceRule::Decider);
    let chi = check_closed(&mesh, "torus");
    assert_eq!(chi, 0, "torus: V − E + F = {chi}, expected 0");
    // With spacing (1, 1, 2) every vertex's z doubles.
    let stretched = marching_cubes(&torus, [1.0, 1.0, 2.0], 0.5, FaceRule::Decider);
    assert_eq!(stretched.vertices.len(), mesh.vertices.len(), "the spacing must not change the number of vertices");
    let zmax = |m: &Mesh| m.vertices.iter().map(|p| p[2]).fold(f32::MIN, f32::max);
    assert!(close(zmax(&stretched), 2.0 * zmax(&mesh), 1e-3), "with Δz = 2 the top of the torus must be twice as high: {} against {}", zmax(&stretched), zmax(&mesh));
}

#[test]
fn r5_cracks_only_with_the_complement_rule_b6() {
    // A noisy volume has many ambiguous faces. A rule that depends only on the face (separate,
    // decider) closes the surface; the complement rule of the 15-case table leaves cracks.
    let mut state = 12345u64;
    let vol = Volume::from_fn(14, 14, 14, |x, y, z| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let border = x == 0 || y == 0 || z == 0 || x == 13 || y == 13 || z == 13;
        if border { 0.0 } else { (state >> 40) as f32 / (1u64 << 24) as f32 }
    });
    for rule in [FaceRule::Separate, FaceRule::Decider] {
        // closed and consistently oriented: every edge is run along as often in one direction as
        // in the other (in noise an edge can belong to 4 triangles, two on each side)
        let mesh = marching_cubes(&vol, [1.0, 1.0, 1.0], 0.5, rule);
        let mut directed: HashMap<(usize, usize), i32> = HashMap::new();
        for t in &mesh.triangles {
            for i in 0..3 {
                *directed.entry((t[i], t[(i + 1) % 3])).or_default() += 1;
            }
        }
        for (&(a, b), &n) in &directed {
            let back = directed.get(&(b, a)).copied().unwrap_or(0);
            assert_eq!(n, back, "noise with {rule:?}: the edge {a}–{b} is run along {n} times one way and {back} times the other (a crack?)");
        }
    }
    let mesh = marching_cubes(&vol, [1.0, 1.0, 1.0], 0.5, FaceRule::Complement);
    let mut undirected: HashMap<(usize, usize), usize> = HashMap::new();
    for t in &mesh.triangles {
        for i in 0..3 {
            let (a, b) = (t[i].min(t[(i + 1) % 3]), t[i].max(t[(i + 1) % 3]));
            *undirected.entry((a, b)).or_default() += 1;
        }
    }
    let cracks = undirected.values().filter(|&&n| n == 1).count();
    assert!(cracks > 0, "B6: with the complement rule, neighbouring cubes should disagree somewhere in a noisy volume: no crack edges found");
}
