// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p marchingcubes surface_measure::

use imgcourse::Volume;
use marchingcubes::surface_measure::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// B1: the octahedron with vertices at distance 0.5 from the centre, triangles anticlockwise
/// seen from outside.
fn octahedron() -> Mesh {
    let vertices = vec![[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]];
    let triangles = vec![[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
    Mesh { vertices, triangles }
}

/// B4(b): the surface of the unit cube, 12 triangles anticlockwise seen from outside.
fn cube() -> Mesh {
    let vertices: Vec<[f32; 3]> = (0..8).map(|k| [(k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32]).collect();
    // each face as 4 corners anticlockwise from outside, split into 2 triangles
    let faces = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
    let triangles = faces.iter().flat_map(|f| [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]).collect();
    Mesh { vertices, triangles }
}

#[test]
fn c1_area_b1() {
    let a = triangle_area([0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]);
    assert!(close(a, 3f32.sqrt() / 8.0, 1e-6), "B1(b): one face of the octahedron: got {a}, expected √3/8 = 0.2165");
    let total = mesh_area(&octahedron());
    assert!(close(total, 3f32.sqrt(), 1e-5), "B1(b): the octahedron: got {total}, expected √3 = 1.732");
    assert!(close(mesh_area(&cube()), 6.0, 1e-5), "the unit cube: got {}, expected 6", mesh_area(&cube()));
}

#[test]
fn c2_volume_b1_b9() {
    let v = mesh_volume(&octahedron());
    assert!(close(v, 1.0 / 6.0, 1e-6), "B1(c): the octahedron: got {v}, expected 1/6");
    assert!(close(mesh_volume(&cube()), 1.0, 1e-5), "the unit cube: got {}, expected 1", mesh_volume(&cube()));
    // B9: moving the mesh does not change its volume.
    let mut moved = cube();
    for p in moved.vertices.iter_mut() {
        *p = [p[0] + 10.0, p[1] - 3.0, p[2] + 7.0];
    }
    assert!(close(mesh_volume(&moved), 1.0, 1e-3), "B9: the cube moved by (10, −3, 7): got {}, expected 1", mesh_volume(&moved));
    // The other orientation gives −V.
    let mut inside_out = cube();
    for t in inside_out.triangles.iter_mut() {
        t.swap(1, 2);
    }
    assert!(close(mesh_volume(&inside_out), -1.0, 1e-5), "triangles turned inside out: got {}, expected −1", mesh_volume(&inside_out));
}

#[test]
fn c3_voxel_estimates_b1_b3() {
    let mut one = Volume::new(3, 3, 3, false);
    one[(1, 1, 1)] = true;
    let (v, a) = voxel_estimates(&one, [1.0, 1.0, 1.0]);
    assert!(close(v, 1.0, 1e-6) && close(a, 6.0, 1e-6), "B1(d): one voxel: got volume {v}, area {a}; expected 1 and 6");
    // B3: the digital disk of radius 3 as a slab one voxel thick: 29 voxels; the exposed faces are
    // the 28 pixel edges of the outline plus the 29 faces on top and the 29 below.
    let disk = Volume::from_fn(7, 7, 1, |x, y, _| (x as i32 - 3).pow(2) + (y as i32 - 3).pow(2) <= 9);
    let (v, a) = voxel_estimates(&disk, [1.0, 1.0, 1.0]);
    assert!(close(v, 29.0, 1e-6), "B3(a): got {v} pixels, expected 29");
    assert!(close(a - 2.0 * 29.0, 28.0, 1e-6), "B3(b): got {} exposed pixel edges, expected 28", a - 58.0);
    // Spacing: voxels of 0.2 × 0.2 × 0.5.
    let (v, a) = voxel_estimates(&one, [0.2, 0.2, 0.5]);
    assert!(close(v, 0.02, 1e-6) && close(a, 2.0 * (0.1 + 0.1 + 0.04), 1e-6), "one voxel of 0.2 × 0.2 × 0.5: got volume {v}, area {a}; expected 0.02 and 0.48");
}

#[test]
fn c4_euler_characteristic_b4() {
    let (v, e, f, b, chi) = topology(&octahedron());
    assert!((v, e, f, b, chi) == (6, 12, 8, 0, 2), "B4(a): octahedron: got V, E, F, boundary, χ = {:?}, expected (6, 12, 8, 0, 2)", (v, e, f, b, chi));
    let (v, e, f, b, chi) = topology(&cube());
    assert!((v, e, f, chi) == (8, 18, 12, 2), "B4(b): cube: got V, E, F, χ = {:?}, expected (8, 18, 12, 2)", (v, e, f, chi));
    assert_eq!(b, 0, "B4(b): the cube is closed");
    let mut holed = octahedron();
    holed.triangles.pop();
    let (_, _, _, b, chi) = topology(&holed);
    assert!(b == 3 && chi == 1, "B4(c): octahedron minus a triangle: got {b} boundary edges and χ = {chi}, expected 3 and 1");
    // B4(e): a torus from a 6 × 4 grid of quads, wrapping round in both directions.
    let (m, n) = (6, 4);
    let id = |i: usize, j: usize| (i % m) * n + (j % n);
    let vertices = vec![[0.0f32; 3]; m * n];
    let triangles = (0..m).flat_map(|i| (0..n).flat_map(move |j| [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]])).collect();
    let (v, e, f, _, chi) = topology(&Mesh { vertices, triangles });
    assert!((v, e, f, chi) == (24, 72, 48, 0), "B4(e): torus: got V, E, F, χ = {:?}, expected (24, 72, 48, 0)", (v, e, f, chi));
}

#[test]
fn all_on_an_extracted_ball() {
    // A ball of radius 6 voxels (value 1 inside, falling to 0 over one voxel at the edge).
    let ball = Volume::from_fn(18, 18, 18, |x, y, z| {
        let d = ((x as f32 - 8.5).powi(2) + (y as f32 - 8.5).powi(2) + (z as f32 - 8.5).powi(2)).sqrt();
        (6.5 - d).clamp(0.0, 1.0)
    });
    let mesh = surface(&ball, [1.0, 1.0, 1.0], 0.5);
    let (a, v) = (mesh_area(&mesh), mesh_volume(&mesh));
    let r: f32 = 6.0;
    assert!(close(a / (4.0 * std::f32::consts::PI * r * r), 1.0, 0.03), "ball of radius 6: area {a:.1}, expected about 4πr² = 452.4");
    assert!(close(v / (4.0 / 3.0 * std::f32::consts::PI * r.powi(3)), 1.0, 0.03), "ball of radius 6: volume {v:.1}, expected about 4/3 πr³ = 904.8");
    let (_, _, _, b, chi) = topology(&mesh);
    assert!(b == 0 && chi == 2, "the ball's surface must be closed (0 boundary edges, χ = 2): got {b} and {chi}");
}
