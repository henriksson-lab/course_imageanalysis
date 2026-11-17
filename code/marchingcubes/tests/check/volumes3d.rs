// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p marchingcubes volumes3d::

use imgcourse::Volume;
use marchingcubes::volumes3d::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const B1: Grid = Grid { spacing: [0.5, 0.5, 2.0], origin: [10.0, 20.0, 5.0] };

#[test]
fn c1_voxel_and_world_b1() {
    let p = voxel_to_world(&B1, [2.0, 3.0, 1.0]);
    let e = [11.0, 21.5, 7.0];
    assert!((0..3).all(|a| close(p[a], e[a], 1e-5)), "B1(b): voxel (2, 3, 1) is at {p:?}, expected {e:?}");
    let v = world_to_voxel(&B1, [12.125, 21.75, 8.5]);
    let e = [4.25, 3.5, 1.75];
    assert!((0..3).all(|a| close(v[a], e[a], 1e-5)), "B1(c): the world point (12.125, 21.75, 8.5) is at voxel {v:?}, expected {e:?}");
    let back = voxel_to_world(&B1, v);
    assert!((0..3).all(|a| close(back[a], [12.125, 21.75, 8.5][a], 1e-4)), "there and back again: got {back:?}");
}

/// A 6 × 5 × 3 volume with the 8 voxel values of B2 around (4.25, 3.5, 1.75), 0 elsewhere.
fn b2_volume() -> Volume<f32> {
    let mut v = Volume::new(6, 5, 3, 0.0f32);
    for (x, y, z, value) in [(4, 3, 1, 10.0), (5, 3, 1, 20.0), (4, 4, 1, 30.0), (5, 4, 1, 60.0), (4, 3, 2, 50.0), (5, 3, 2, 50.0), (4, 4, 2, 90.0), (5, 4, 2, 70.0)] {
        v[(x, y, z)] = value;
    }
    v
}

#[test]
fn c2_trilinear_b2() {
    let v = b2_volume();
    let got = trilinear(&v, [4.25, 3.5, 1.75]);
    assert!(close(got, 56.875, 1e-3), "B2(a): got {got}, expected 56.875");
    let centre = trilinear(&v, [4.5, 3.5, 1.5]);
    assert!(close(centre, 47.5, 1e-3), "B2(c): the centre of the cube: got {centre}, expected the mean 47.5");
    // On a voxel, exactly its value; also on the last voxel of every axis.
    assert!(close(trilinear(&v, [4.0, 4.0, 2.0]), 90.0, 1e-5), "at voxel (4, 4, 2): got {}, expected 90", trilinear(&v, [4.0, 4.0, 2.0]));
    assert!(close(trilinear(&v, [5.0, 4.0, 2.0]), 70.0, 1e-5), "at the last voxel (5, 4, 2): got {}, expected 70", trilinear(&v, [5.0, 4.0, 2.0]));
    assert_eq!(trilinear(&v, [5.5, 2.0, 1.0]), 0.0, "outside the volume the value must be 0");
    assert_eq!(trilinear(&v, [2.0, 2.0, -0.5]), 0.0, "outside the volume the value must be 0");
}

#[test]
fn c3_oblique_slice_b6() {
    // A volume whose value is i + 10 j + 100 k: trilinear interpolation reproduces it exactly,
    // so every output pixel tells where it sampled.
    let vol = Volume::from_fn(40, 45, 20, |i, j, k| i as f32 + 10.0 * j as f32 + 100.0 * k as f32);
    let grid = Grid { spacing: [0.2, 0.2, 0.5], origin: [0.0, 0.0, 0.0] };
    let w = [0.0, 30f32.to_radians().cos(), 30f32.to_radians().sin()];
    let slice = oblique_slice(&vol, &grid, [6.0, 6.0, 5.0], [1.0, 0.0, 0.0], w, 0.2, 21, 41);
    // Output pixel (p, q) = (5, 10) relative to the centre is (15, 30) in a 21 × 41 slice.
    let got = slice[(15, 30)];
    let expected = 35.0 + 10.0 * 38.66 + 100.0 * 12.0;
    assert!(close(got, expected, 0.1), "B6: pixel (p, q) = (5, 10) sampled the value {got}, expected {expected} (voxel (35, 38.66, 12))");
    let centre = slice[(10, 20)];
    assert!(close(centre, 30.0 + 300.0 + 1000.0, 0.01), "the centre pixel must sample voxel (30, 30, 10): got {centre}");
    // An axis-aligned plane is an ordinary xy slice.
    let xy = oblique_slice(&vol, &grid, [3.8, 4.4, 2.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.2, 39, 45);
    assert!(close(xy[(0, 0)], vol[(0, 0, 5)], 1e-3) && close(xy[(20, 7)], vol[(20, 7, 5)], 1e-3), "a plane along x and y through slice 5 must be slice 5");
}

/// B3: three 5 × 5 slices.
fn b3_stack() -> Volume<f32> {
    let slices = [
        [[0, 0, 0, 0, 0], [0, 9, 9, 0, 0], [0, 9, 9, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 7], [0, 0, 0, 7, 7]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 8, 8, 0], [0, 0, 8, 8, 0], [0, 0, 0, 0, 0]],
    ];
    Volume::from_fn(5, 5, 3, |x, y, z| slices[z][y][x] as f32)
}

#[test]
fn c4_mip_b3() {
    let mip = mip_z(&b3_stack());
    let expected = [[0., 0., 0., 0., 0.], [0., 9., 9., 0., 0.], [0., 9., 9., 8., 0.], [0., 0., 8., 8., 7.], [0., 0., 0., 7., 7.]];
    for (x, y) in mip.coords() {
        assert_eq!(mip[(x, y)], expected[y][x], "B3(a): MIP at row {y}, column {x}: got {}, expected {}", mip[(x, y)], expected[y][x]);
    }
}

#[test]
fn c5_counting_objects_b3() {
    let stack = b3_stack();
    let mask = stack.map(|v| v > 5.0);
    let (labels6, n6) = label_3d(&mask, 6);
    let (_, n26) = label_3d(&mask, 26);
    assert_eq!(n6, 3, "B3(c): 6-connected objects: got {n6}, expected 3");
    assert_eq!(n26, 2, "B3(c): 26-connected objects: got {n26}, expected 2");
    assert_ne!(labels6[(1, 1, 0)], labels6[(2, 2, 2)], "the 9-block and the 8-block are different objects");
    assert_eq!(labels6[(0, 0, 0)], 0, "background must be 0");
    // The thresholded MIP as a volume of depth 1: 26-connected = 8-connected in 2D.
    let mip = Volume::from_fn(5, 5, 1, |x, y, _| (0..3).any(|z| stack[(x, y, z)] > 5.0));
    let (_, n_mip) = label_3d(&mip, 26);
    assert_eq!(n_mip, 1, "B3(b): objects in the MIP: got {n_mip}, expected 1");
}
