// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p unmixing lsunmix::

use imgcourse::Image;
use unmixing::lsunmix::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn b1() -> Matrix {
    vec![vec![1.0, 0.2], vec![0.5, 0.5], vec![0.1, 1.0]]
}

#[test]
fn c1_unmix_b1() {
    let a = unmix_pixel(&b1(), &[1.3, 0.9, 1.2]);
    assert!(close(a[0], 1.029, 1e-3) && close(a[1], 1.042, 1e-3), "B1(b): got {a:?}, expected (1.029, 1.042)");
}

#[test]
fn c1_unmix_b4() {
    let m = vec![vec![1.0, 0.02], vec![0.45, 0.30], vec![0.10, 1.0], vec![0.02, 0.40]];
    let a = unmix_pixel(&m, &[104.0, 58.0, 61.0, 24.0]);
    assert!(close(a[0], 101.71, 0.01) && close(a[1], 50.65, 0.01), "B4(b): got {a:?}, expected (101.71, 50.65)");
}

#[test]
fn c2_pseudo_inverse_b2() {
    let near = vec![vec![1.0, 0.9], vec![0.9, 1.0], vec![0.5, 0.5]];
    let p = pseudo_inverse(&near);
    assert!(p.len() == 2 && p[0].len() == 3, "M⁺ of a 3 × 2 matrix must be 2 × 3, got {} × {}", p.len(), p[0].len());
    assert!(close(p[0][0], 5.23, 0.01) && close(p[1][0], -4.77, 0.01), "B2(c), first column: got ({}, {}), expected (5.23, −4.77)", p[0][0], p[1][0]);
    let sep = vec![vec![1.0, 0.1], vec![0.1, 1.0], vec![0.5, 0.5]];
    let p = pseudo_inverse(&sep);
    assert!(close(p[0][0], 0.88, 0.01) && close(p[1][0], -0.23, 0.01), "B2(c), separated: got ({}, {}), expected (0.88, −0.23)", p[0][0], p[1][0]);
    // M⁺ M = I
    for i in 0..2 {
        for j in 0..2 {
            let v: f32 = (0..3).map(|c| p[i][c] * sep[c][j]).sum();
            assert!(close(v, if i == j { 1.0 } else { 0.0 }, 1e-4), "M⁺M at ({i}, {j}): got {v}");
        }
    }
}

#[test]
fn c2_unmix_image_matches_pixel() {
    let m = b1();
    let stack = vec![Image::from_rows(&[[1.3f32, 1.2]]), Image::from_rows(&[[0.9f32, 1.0]]), Image::from_rows(&[[1.2f32, 1.1]])];
    let maps = unmix_image(&stack, &m);
    assert!(maps.len() == 2, "two components must give two maps, got {}", maps.len());
    assert!(close(maps[0][(0, 0)], 1.029, 1e-3) && close(maps[1][(0, 0)], 1.042, 1e-3), "pixel 0 (B1): got ({}, {})", maps[0][(0, 0)], maps[1][(0, 0)]);
    // pixel 1 is the exact mixture (1, 1)
    assert!(close(maps[0][(1, 0)], 1.0, 1e-4) && close(maps[1][(1, 0)], 1.0, 1e-4), "an exact mixture (1, 1): got ({}, {})", maps[0][(1, 0)], maps[1][(1, 0)]);
}

#[test]
fn c3_residual_b1c() {
    let m = b1();
    let y = [1.3, 0.9, 1.2];
    let r = residual(&m, &y, &unmix_pixel(&m, &y));
    let e = [0.062, -0.136, 0.055];
    assert!((0..3).all(|c| close(r[c], e[c], 1e-3)), "B1(c): got {r:?}, expected {e:?}");
    for k in 0..2 {
        let dot: f32 = (0..3).map(|c| m[c][k] * r[c]).sum();
        assert!(dot.abs() < 1e-4, "the residual must be orthogonal to column {}: dot product {dot}", k + 1);
    }
}

#[test]
fn c3_residual_map() {
    // an exact mixture has residual 0; a pixel off the plane does not
    let m = b1();
    let stack = vec![Image::from_rows(&[[12.0f32, 13.0]]), Image::from_rows(&[[10.0f32, 9.0]]), Image::from_rows(&[[11.0f32, 12.0]])];
    let maps = unmix_image(&stack, &m);
    let res = residual_map(&stack, &m, &maps);
    assert!(res[(0, 0)] < 1e-3, "the exact mixture (10, 10): got residual {}, expected 0", res[(0, 0)]);
    assert!(res[(1, 0)] > 0.1, "a pixel off the plane: got residual {}, expected more than 0", res[(1, 0)]);
}

#[test]
fn c4_condition_b2() {
    let near = vec![vec![1.0, 0.9], vec![0.9, 1.0], vec![0.5, 0.5]];
    let sep = vec![vec![1.0, 0.1], vec![0.1, 1.0], vec![0.5, 0.5]];
    let (kn, ks) = (condition_number(&near), condition_number(&sep));
    assert!(close(kn, 20.3, 0.1), "B2(b), near-parallel: got κ = {kn}, expected 20.3");
    assert!(close(ks, 1.45, 0.01), "B2(b), separated: got κ = {ks}, expected 1.45");
    let id = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
    assert!(close(condition_number(&id), 1.0, 1e-5), "2 I: got κ = {}, expected 1", condition_number(&id));
}

#[test]
fn c5_weighted() {
    let m = b1();
    let y = [1.3, 0.9, 1.2];
    let a = unmix_weighted(&m, &y, &[2.0, 2.0, 2.0]);
    assert!(close(a[0], 1.029, 1e-3) && close(a[1], 1.042, 1e-3), "equal weights must give ordinary LS (B1): got {a:?}");
    // exact data: any weights give the exact answer
    let a = unmix_weighted(&m, &[1.2, 1.0, 1.1], &[1.0, 0.01, 5.0]);
    assert!(close(a[0], 1.0, 1e-4) && close(a[1], 1.0, 1e-4), "an exact mixture, unequal weights: got {a:?}, expected (1, 1)");
    // a huge weight on channels 1 and 3 makes them fit (almost) exactly
    let a = unmix_weighted(&m, &y, &[1e4, 1e-4, 1e4]);
    let r = residual(&m, &y, &a);
    assert!(r[0].abs() < 1e-3 && r[2].abs() < 1e-3, "channels with a huge weight must fit: residual {r:?}");
}
