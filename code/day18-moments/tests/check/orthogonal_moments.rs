// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day18 orthogonal_moments::

use imgcourse::Image;
use day18::orthogonal_moments::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_legendre_b8() {
    let cases = [(2, 0.5, -0.125), (3, 0.5, -0.4375), (4, 0.5, -0.2890625), (2, 0.0, -0.5), (4, 0.0, 0.375)];
    for (n, x, expected) in cases {
        let got = legendre(n, x);
        assert!(close(got, expected, 1e-12), "P{n}({x}): got {got}, expected {expected}");
    }
    for n in 0..12 {
        assert!(close(legendre(n, 1.0), 1.0, 1e-12), "P{n}(1): got {}, expected 1", legendre(n, 1.0));
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        assert!(close(legendre(n, -1.0), sign, 1e-12), "P{n}(-1): got {}, expected {sign}", legendre(n, -1.0));
    }
}

#[test]
fn c2_gram_schmidt_b1() {
    // on the three points -1, 0, 1: 1, x, x² - 2/3, each scaled to length 1
    let t = gram_schmidt(2, &[-1.0, 0.0, 1.0]);
    let s3 = 3f64.sqrt();
    let expected = [[1.0 / s3, 1.0 / s3, 1.0 / s3], [-1.0 / 2f64.sqrt(), 0.0, 1.0 / 2f64.sqrt()], [1.0 / 6f64.sqrt(), -2.0 / 6f64.sqrt(), 1.0 / 6f64.sqrt()]];
    for n in 0..3 {
        for i in 0..3 {
            assert!(close(t[n][i], expected[n][i], 1e-12), "vector {n}: got {:?}, expected {:?}", t[n], expected[n]);
        }
    }
}

#[test]
fn c2_gram_schmidt_orthonormal() {
    let xs: Vec<f64> = (0..16).map(|i| centre(i, 16)).collect();
    let t = gram_schmidt(10, &xs);
    for m in 0..=10 {
        for n in 0..=10 {
            let dot: f64 = t[m].iter().zip(&t[n]).map(|(a, b)| a * b).sum();
            let expected = if m == n { 1.0 } else { 0.0 };
            assert!(close(dot, expected, 1e-9), "vectors {m} and {n} on 16 pixels: dot product {dot:.2e}, expected {expected}");
        }
    }
}

#[test]
fn c3_constant_image_b7() {
    // 4 × 4 ones: the sampled P0 and P2 are not orthogonal on 4 pixels, so a20 = 5/4 · (-1/16) · 2 = -5/32
    let a = legendre_moments(&Image::new(4, 4, 1.0), 2);
    let cases = [("a00", a[0][0], 1.0), ("a10", a[1][0], 0.0), ("a11", a[1][1], 0.0), ("a20", a[2][0], -5.0 / 32.0), ("a02", a[0][2], -5.0 / 32.0)];
    for (name, got, expected) in cases {
        assert!(close(got, expected, 1e-12), "{name}: got {got}, expected {expected}");
    }
}

#[test]
fn c3_absolute_value_b3() {
    // f(x) = |x| on 400 samples (an image 400 wide, 1 high): a0 = 1/2, a1 = 0, a2 = 5/8
    let img = Image::from_fn(400, 1, |x, _| centre(x, 400).abs() as f32);
    let a = legendre_moments(&img, 2);
    let cases = [("a0", a[0][0], 0.5), ("a1", a[1][0], 0.0), ("a2", a[2][0], 0.625)];
    for (name, got, expected) in cases {
        assert!(close(got, expected, 1e-4), "{name} of |x|: got {got:.5}, expected {expected}");
    }
}

#[test]
fn c4_reconstruct_a_polynomial() {
    // a polynomial of degree 2 is rebuilt from its moments up to order 2
    let (w, h) = (100, 80);
    let f = |x: usize, y: usize| {
        let (u, v) = (centre(x, w), centre(y, h));
        (0.3 + 0.2 * u - 0.1 * u * v + 0.4 * v * v) as f32
    };
    let img = Image::from_fn(w, h, f);
    let rec = reconstruct(&legendre_moments(&img, 2), w, h);
    for (x, y) in img.coords() {
        assert!((rec[(x, y)] - img[(x, y)]).abs() < 1e-3, "pixel ({x}, {y}): rebuilt {}, original {}", rec[(x, y)], img[(x, y)]);
    }
    // order 0: the mean everywhere
    let flat = reconstruct(&legendre_moments(&img, 0), w, h);
    assert!((flat[(7, 9)] - img.mean()).abs() < 1e-4, "order 0: got {}, expected the mean {}", flat[(7, 9)], img.mean());
}
