// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p compression dct::

use compression::dct::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_matrix_4_points() {
    let d = dct_matrix(4);
    let expected = [[0.5, 0.5, 0.5, 0.5], [0.653, 0.271, -0.271, -0.653], [0.5, -0.5, -0.5, 0.5], [0.271, -0.653, 0.653, -0.271]];
    for k in 0..4 {
        for n in 0..4 {
            assert!(close(d[k][n], expected[k][n], 1e-3), "row {k}, column {n}: got {:.3}, expected {}", d[k][n], expected[k][n]);
        }
    }
}

#[test]
fn c1_matrix_orthonormal_b2() {
    let d = dct_matrix(8);
    for i in 0..8 {
        for j in 0..8 {
            let dot: f32 = (0..8).map(|n| d[i][n] * d[j][n]).sum();
            let e = if i == j { 1.0 } else { 0.0 };
            assert!(close(dot, e, 1e-5), "rows {i} and {j} of the 8-point DCT: dot product {dot}, expected {e}");
        }
    }
}

#[test]
fn c2_two_points_b1() {
    let c = dct(&[100.0, 104.0]);
    assert!(close(c[0], 144.25, 0.01) && close(c[1], -2.83, 0.01), "B1(a): got {c:?}, expected [144.25, -2.83]");
    let x = idct(&[c[0], 0.0]);
    assert!(close(x[0], 102.0, 1e-3) && close(x[1], 102.0, 1e-3), "B1(b), c1 dropped: got {x:?}, expected [102, 102]");
}

#[test]
fn c2_four_points_b2_b3() {
    let c = dct(&[1.0, 1.0, 1.0, 1.0]);
    assert!(close(c[0], 2.0, 1e-5) && c[1..].iter().all(|v| v.abs() < 1e-5), "B2(a), (1, 1, 1, 1): got {c:?}, expected [2, 0, 0, 0]");
    let c = dct(&[1.0, 2.0, 3.0, 4.0]);
    let expected = [5.0, -2.230, 0.0, -0.159];
    for k in 0..4 {
        assert!(close(c[k], expected[k], 1e-3), "B2(a), (1, 2, 3, 4): got {c:?}, expected {expected:?}");
    }
    let energy: f32 = c.iter().map(|v| v * v).sum();
    assert!(close(energy, 30.0, 1e-3), "B2(c), Parseval: sum of squared coefficients {energy}, expected 30");
    let x = idct(&[c[0], 0.0, 0.0, 0.0]);
    assert!(x.iter().all(|&v| close(v, 2.5, 1e-4)), "B3, only c0 kept: got {x:?}, expected the mean 2.5 everywhere");
    let x = idct(&c);
    assert!(x.iter().zip([1.0, 2.0, 3.0, 4.0]).all(|(&a, b)| close(a, b, 1e-4)), "idct(dct(x)) should give x back: got {x:?}");
}

#[test]
fn c3_flat_block_b5a() {
    let block = Image::new(8, 8, 200.0 - 128.0);
    let c = dct2(&block);
    assert!(close(c[(0, 0)], 576.0, 1e-2), "B5(a): C00 = {}, expected 576", c[(0, 0)]);
    for (u, v) in c.coords().skip(1) {
        assert!(c[(u, v)].abs() < 1e-3, "B5(a): coefficient ({u}, {v}) = {}, expected 0", c[(u, v)]);
    }
}

#[test]
fn c3_vertical_edge_b5b() {
    // left four columns 200, right four 60, minus 128
    let block = Image::from_fn(8, 8, |x, _| if x < 4 { 72.0 } else { -68.0 });
    let c = dct2(&block);
    let top = [16.0, 507.4, 0.0, -178.2, 0.0, 119.1, 0.0, -100.9]; // the demo's numbers (D1)
    for u in 0..8 {
        assert!(close(c[(u, 0)], top[u], 0.1), "B5(b), top row u = {u}: got {:.1}, expected {}", c[(u, 0)], top[u]);
    }
    for (u, v) in c.coords().filter(|&(_, v)| v > 0) {
        assert!(c[(u, v)].abs() < 1e-3, "B5(b): coefficient (u, v) = ({u}, {v}) should be 0 (only the top row), got {}", c[(u, v)]);
    }
}

#[test]
fn c3_inverse_and_parseval() {
    let block = samples::cell().crop(112, 96, 8, 8);
    let c = dct2(&block);
    let back = idct2(&c);
    for (x, y) in block.coords() {
        assert!(close(back[(x, y)], block[(x, y)], 1e-4), "pixel ({x}, {y}): idct2(dct2) gave {}, expected {}", back[(x, y)], block[(x, y)]);
    }
    let e1: f32 = block.pixels().iter().map(|v| v * v).sum();
    let e2: f32 = c.pixels().iter().map(|v| v * v).sum();
    assert!(close(e1, e2, 1e-3), "Parseval: pixel energy {e1}, coefficient energy {e2}");
}

#[test]
fn c4_keep_all_and_keep_one_per_block() {
    let img = samples::cell().crop(0, 0, 32, 16);
    let all = keep_largest_blocks(&img, 1.0);
    for (x, y) in img.coords() {
        assert!(close(all[(x, y)], img[(x, y)], 1e-4), "keeping 100 %: pixel ({x}, {y}) is {}, expected {}", all[(x, y)], img[(x, y)]);
    }
    // A flat image: each block has only its DC. Keeping 1/64 keeps exactly those.
    let flat = Image::from_fn(16, 16, |x, y| if x < 8 && y < 8 { 0.2 } else { 0.7 });
    let out = keep_largest_blocks(&flat, 1.0 / 64.0);
    for (x, y) in flat.coords() {
        assert!(close(out[(x, y)], flat[(x, y)], 1e-4), "flat blocks, 1/64 kept: pixel ({x}, {y}) is {}, expected {}", out[(x, y)], flat[(x, y)]);
    }
}
