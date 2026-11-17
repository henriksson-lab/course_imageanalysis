// A three-layer flow with fixed (hand-chosen) coupling functions that bends a Gaussian into a
// banana. Checks the inverse, the log-determinant against a numerical Jacobian, that the density
// integrates to 1, and the change-of-variables density against a histogram of 200 000 samples.
// Pictures in out/flows/.
//   cargo run -p gans -- flows                      (your code)
//   cargo run -p gans --features solution -- flows  (the model solution)

use crate::flows::*;
use imgcourse::*;

/// The flow of the program (data → noise).
pub fn banana() -> Vec<Coupling> {
    vec![
        Coupling { d: 1, s: |x0| 0.3 * x0.tanh(), t: |x0| 0.5 - 0.25 * x0 * x0 },
        Coupling { d: 0, s: |_| 0.8f64.ln(), t: |x1| 0.3 * x1 },
        Coupling { d: 1, s: |_| 0.0, t: |x0| 0.2 * x0 },
    ]
}

fn forward_all(flow: &[Coupling], x: [f64; 2]) -> ([f64; 2], f64) {
    let mut v = x;
    let mut ld = 0.0;
    for l in flow {
        let (y, s) = coupling_forward(l, v);
        v = y;
        ld += s;
    }
    (v, ld)
}

pub fn main() {
    let dir = "out/flows";
    let flow = banana();
    // the by-hand layer of exercise B3: s = ln 2, t = 1, changing coordinate 1
    let b3 = Coupling { d: 1, s: |_| 2f64.ln(), t: |_| 1.0 };
    let (y, s) = coupling_forward(&b3, [0.5, 1.0]);
    println!("B3: (0.5, 1) -> ({:.3}, {:.3}), log det {:.4}; back: {:?}", y[0], y[1], s, coupling_inverse(&b3, y));

    // inverse and log-determinant
    let mut rng = Rng::new(2);
    let mut worst: f64 = 0.0;
    for _ in 0..1000 {
        let x = sample(&flow, &mut rng);
        let (z, _) = forward_all(&flow, x);
        let back = flow.iter().rev().fold(z, |v, l| coupling_inverse(l, v));
        worst = worst.max((back[0] - x[0]).abs().max((back[1] - x[1]).abs()));
    }
    println!("largest error of inverse(forward(x)) over 1000 samples: {worst:.1e}");
    let x = [0.7, -0.4];
    let h = 1e-6;
    let f = |p: [f64; 2]| forward_all(&flow, p).0;
    let (a, b, c, d) = (
        (f([x[0] + h, x[1]])[0] - f([x[0] - h, x[1]])[0]) / (2.0 * h),
        (f([x[0], x[1] + h])[0] - f([x[0], x[1] - h])[0]) / (2.0 * h),
        (f([x[0] + h, x[1]])[1] - f([x[0] - h, x[1]])[1]) / (2.0 * h),
        (f([x[0], x[1] + h])[1] - f([x[0], x[1] - h])[1]) / (2.0 * h),
    );
    println!("at x = (0.7, -0.4): sum of s = {:.5}, log |det| of the numerical Jacobian = {:.5}", forward_all(&flow, x).1, (a * d - b * c).abs().ln());

    // density on a grid: integral, and against a histogram
    let mut integral = 0.0;
    let fw = 0.02;
    for i in 0..700 {
        for j in 0..850 {
            integral += log_density(&flow, [-7.0 + (i as f64 + 0.5) * fw, -5.0 + (j as f64 + 0.5) * fw]).exp() * fw * fw;
        }
    }
    println!("integral of p(x) over [-7, 7] x [-5, 12]: {integral:.4}");
    let (x0, y0, n) = (-4.0, -3.0, 40usize);
    let w = 0.2;
    let m = 200_000;
    let mut hist = vec![0.0; n * n];
    let mut rng = Rng::new(1);
    for _ in 0..m {
        let p = sample(&flow, &mut rng);
        let (i, j) = (((p[0] - x0) / w).floor(), ((p[1] - y0) / w).floor());
        if i >= 0.0 && j >= 0.0 && (i as usize) < n && (j as usize) < n {
            hist[j as usize * n + i as usize] += 1.0 / m as f64;
        }
    }
    let mut tv = 0.0;
    let mut dens_img = Image::new(n, n, 0.0f32);
    let mut hist_img = Image::new(n, n, 0.0f32);
    let mut biggest = (0.0, 0.0);
    for j in 0..n {
        for i in 0..n {
            let centre = [x0 + (i as f64 + 0.5) * w, y0 + (j as f64 + 0.5) * w];
            let expected = log_density(&flow, centre).exp() * w * w;
            tv += 0.5 * (hist[j * n + i] - expected).abs();
            if hist[j * n + i] > biggest.0 {
                biggest = (hist[j * n + i], expected);
            }
            // y up in the picture
            dens_img[(i, n - 1 - j)] = expected as f32;
            hist_img[(i, n - 1 - j)] = hist[j * n + i] as f32;
        }
    }
    println!("histogram of {m} samples in {n} x {n} bins of 0.2 x 0.2 on [-4, 4] x [-3, 5] against p(x) x bin area:");
    println!("  fullest bin: {:.4} of the samples, p(x) x area there {:.4}", biggest.0, biggest.1);
    println!("  half the summed absolute difference (0 = identical): {tv:.4}");
    save_gray_stretched(&dens_img, format!("{dir}/density.png"));
    save_gray_stretched(&hist_img, format!("{dir}/histogram.png"));
    println!("wrote {dir}/density.png and {dir}/histogram.png");
}
