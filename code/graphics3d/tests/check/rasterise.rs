// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p graphics3d rasterise::

use imgcourse::Image;
use graphics3d::rasterise::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

const A: Vec2 = [1.0, 1.0];
const B: Vec2 = [7.0, 2.0];
const C: Vec2 = [3.0, 6.0];

#[test]
fn c1_edge_functions_b1() {
    assert!(close(edge(A, B, C), 28.0, 1e-12), "E_ab(c): got {}, expected 28", edge(A, B, C));
    // pixel centre, E_bc, E_ca, E_ab (B1 b)
    let table = [([3.5, 2.5], [12.0, 9.5, 6.5]), ([6.5, 4.5], [-8.0, 20.5, 15.5]), ([2.5, 5.5], [4.0, -1.5, 25.5]), ([4.5, 3.5], [4.0, 12.5, 11.5])];
    for (p, e) in table {
        let got = [edge(B, C, p), edge(C, A, p), edge(A, B, p)];
        for k in 0..3 {
            assert!(close(got[k], e[k], 1e-12), "at {p:?}: got E_bc, E_ca, E_ab = {got:?}, expected {e:?}");
        }
    }
}

#[test]
fn c2_barycentric_b1_b4() {
    let b = barycentric(A, B, C, [4.5, 3.5]).expect("(4.5, 3.5) is inside");
    let expected = [4.0 / 28.0, 12.5 / 28.0, 11.5 / 28.0];
    for k in 0..3 {
        assert!(close(b[k], expected[k], 1e-12), "barycentrics of (4.5, 3.5): got {b:?}, expected (0.143, 0.446, 0.411)");
    }
    let z = b[0] * 0.2 + b[1] * 0.6 + b[2] * 0.4;
    assert!(close(z, 0.461, 1e-3), "interpolated depth: got {z}, expected 0.461");
    assert!(barycentric(A, B, C, [6.5, 4.5]).is_none(), "(6.5, 4.5) is outside (beyond edge bc)");
    assert!(barycentric(A, B, C, [2.5, 5.5]).is_none(), "(2.5, 5.5) is outside (beyond edge ca)");
    // B4: the same triangle in the other order is clockwise: a back face, culled
    assert!(barycentric([0., 0.], [4., 0.], [0., 3.], [1.0, 1.0]).is_some(), "(0, 0), (4, 0), (0, 3) is counter-clockwise: (1, 1) must be inside");
    assert!(barycentric([0., 0.], [0., 3.], [4., 0.], [1.0, 1.0]).is_none(), "(0, 0), (0, 3), (4, 0) is clockwise: it must be culled");
}

/// Draw the two triangles of B2 into a 4 × 4 buffer, in the given order. Colour 1 = T1, 2 = T2.
fn b2(t1_first: bool) -> (Image<f32>, Image<f32>, [usize; 2]) {
    let mut colour = Image::new(4, 4, 0.0f32);
    let mut depth = Image::new(4, 4, f32::INFINITY);
    let t1 = [[0.0, 0.0, 0.3], [4.5, 0.0, 0.3], [0.0, 4.5, 0.3]];
    let t2 = [[0.0, 1.0, 0.6], [4.0, 1.0, 0.1], [4.0, 4.0, 0.1]];
    let mut n = [0, 0];
    if t1_first {
        n[0] = draw_triangle(&mut colour, &mut depth, t1, &|_| 1.0);
        n[1] = draw_triangle(&mut colour, &mut depth, t2, &|_| 2.0);
    } else {
        n[1] = draw_triangle(&mut colour, &mut depth, t2, &|_| 2.0);
        n[0] = draw_triangle(&mut colour, &mut depth, t1, &|_| 1.0);
    }
    (colour, depth, n)
}

#[test]
fn c3_z_buffer_b2() {
    // B2 (c), rows from j = 0 (bottom) to 3; 0 = empty
    let want_colour = [[1., 1., 1., 1.], [1., 1., 2., 2.], [1., 1., 2., 2.], [1., 0., 0., 2.]];
    let want_depth = [[0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.2875, 0.1625], [0.3, 0.3, 0.2875, 0.1625], [0.3, f32::INFINITY, f32::INFINITY, 0.1625]];
    for order in [true, false] {
        let (colour, depth, _) = b2(order);
        for j in 0..4 {
            for i in 0..4 {
                assert!(colour[(i, j)] == want_colour[j][i], "drawing {} first, pixel ({i}, {j}): got triangle {}, expected {}", if order { "T1" } else { "T2" }, colour[(i, j)], want_colour[j][i]);
                let (g, e) = (depth[(i, j)], want_depth[j][i]);
                assert!(g == e || (g - e).abs() < 1e-6, "drawing {} first, depth at ({i}, {j}): got {g}, expected {e}", if order { "T1" } else { "T2" });
            }
        }
    }
    // pixel counts when drawn into empty buffers: T1 covers 10, T2 covers 6 (B2 a)
    let (_, _, n) = b2(true);
    assert_eq!(n[0], 10, "T1 drawn first into an empty buffer: got {} pixels, expected 10", n[0]);
}

#[test]
fn c4_perspective_correct_b5() {
    // an edge from w = 1 (u = 0) to w = 3 (u = 1): the third corner gets weight 0
    let mid = perspective_correct([0.5, 0.5, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 0.0]);
    assert!(close(mid, 0.25, 1e-12), "u at the screen midpoint: got {mid}, expected 0.25");
    let s = 0.75;
    let u = perspective_correct([1.0 - s, s, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 0.0]);
    assert!(close(u, 0.5, 1e-12), "u three quarters along the edge: got {u}, expected 0.5");
    // with equal w it is ordinary interpolation
    let u = perspective_correct([0.2, 0.3, 0.5], [2.0, 2.0, 2.0], [1.0, 2.0, 3.0]);
    assert!(close(u, 2.3, 1e-12), "equal w: got {u}, expected 2.3");
}

#[test]
fn c5_shading_b7() {
    let n = [0.0, 0.0, 1.0];
    let v = [0.0, 0.0, 1.0];
    let s2 = 0.5f64.sqrt();
    let l = [s2, 0.0, s2];
    let lambert = phong(n, l, v, 0.1, 0.8, 0.0, 20.0);
    assert!(close(lambert, 0.666, 1e-3), "Lambert part (ks = 0): got {lambert}, expected 0.666");
    let p = phong(n, l, v, 0.1, 0.8, 0.5, 20.0) - lambert;
    assert!(close(p, 0.00049, 2e-5), "Phong specular term: got {p}, expected 0.00049");
    let b = blinn_phong(n, l, v, 0.1, 0.8, 0.5, 20.0) - lambert;
    assert!(close(b, 0.103, 1e-3), "Blinn–Phong specular term: got {b}, expected 0.103");
    // light from behind: only the ambient part
    let back = blinn_phong(n, [0.0, 0.0, -1.0], v, 0.1, 0.8, 0.5, 20.0);
    assert!(close(back, 0.1, 1e-12), "light from behind the surface: got {back}, expected 0.1");
}
