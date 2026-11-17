// Rebuilds three 64 × 64 images from their Legendre moments up to order N; results in
// out/orthogonal-moments/.
//   cargo run -p day18 -- orthogonal-moments                      (your code)
//   cargo run -p day18 --features solution -- orthogonal-moments  (the model solution)

use imgcourse::*;
use crate::orthogonal_moments::*;

/// Relative error ||rebuilt - original|| / ||original||.
fn rel_error(a: &Image<f32>, b: &Image<f32>) -> f64 {
    let num: f64 = a.coords().map(|(x, y)| ((a[(x, y)] - b[(x, y)]) as f64).powi(2)).sum();
    let den: f64 = a.pixels().iter().map(|&v| (v as f64).powi(2)).sum();
    (num / den).sqrt()
}

/// The same projection with the discrete orthogonal polynomials of `gram_schmidt` (exactly
/// orthogonal on the pixels): coefficients `sum of T_p(x) T_q(y) I`, then the double sum.
fn discrete_reconstruct(img: &Image<f32>, order: usize) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    let tx = gram_schmidt(order, &(0..w).map(|i| centre(i, w)).collect::<Vec<_>>());
    let ty = gram_schmidt(order, &(0..h).map(|i| centre(i, h)).collect::<Vec<_>>());
    let mut a = vec![vec![0.0; order + 1]; order + 1];
    for p in 0..=order {
        for q in 0..=order - p {
            a[p][q] = img.coords().map(|(x, y)| tx[p][x] * ty[q][y] * img[(x, y)] as f64).sum();
        }
    }
    Image::from_fn(w, h, |x, y| {
        let mut s = 0.0;
        for p in 0..=order {
            for q in 0..=order - p {
                s += a[p][q] * tx[p][x] * ty[q][y];
            }
        }
        s as f32
    })
}

pub fn main() {
    let dir = "out/orthogonal-moments";
    // a bean-shaped nucleus: an ellipse with a dent on one side
    let nucleus = Image::from_fn(64, 64, |x, y| {
        let (u, v) = ((x as f32 - 31.5) / 22.0, (y as f32 - 31.5) / 15.0);
        let dent = (x as f32 - 31.5).hypot(y as f32 - 52.0) < 11.0;
        if u * u + v * v <= 1.0 && !dent { 0.8 } else { 0.1 }
    });
    let images = [("nucleus", nucleus), ("letter F", samples::letter_f(64)), ("micrograph", samples::cell().crop(96, 96, 64, 64))];
    let orders = [2, 8, 20, 40];
    let mut sheet = Image::new(5 * 66, 3 * 66, 0.5f32);
    println!("relative error of the reconstruction from the Legendre moments up to order N:");
    println!("{:>12} {}", "N =", orders.map(|n| format!("{n:>7}")).join(""));
    for (row, (name, img)) in images.iter().enumerate() {
        let mut line = String::new();
        for (col, &n) in orders.iter().enumerate() {
            let rec = reconstruct(&legendre_moments(img, n), 64, 64);
            line += &format!("{:7.3}", rel_error(img, &rec));
            for (x, y) in rec.coords() {
                sheet[((col + 1) * 66 + x, row * 66 + y)] = rec[(x, y)];
            }
        }
        for (x, y) in img.coords() {
            sheet[(x, row * 66 + y)] = img[(x, y)];
        }
        println!("{name:>12} {line}");
    }
    let n = 40;
    println!("(each order N uses (N+1)(N+2)/2 numbers: {})", orders.map(|n| ((n + 1) * (n + 2) / 2).to_string()).join(", "));
    let cell = &images[2].1;
    println!("micrograph, N = {n}: sampled Legendre {:.3}, discrete orthogonal polynomials {:.3}", rel_error(cell, &reconstruct(&legendre_moments(cell, n), 64, 64)), rel_error(cell, &discrete_reconstruct(cell, n)));
    let big = Image::from_fn(sheet.width() * 2, sheet.height() * 2, |x, y| sheet[(x / 2, y / 2)]);
    save_gray(&big, format!("{dir}/reconstructions.png"));
    println!("wrote {dir}/reconstructions.png (rows: the three images; columns: input, N = 2, 8, 20, 40)");
}
