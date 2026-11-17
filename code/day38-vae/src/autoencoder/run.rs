// PCA as an autoencoder on the micrograph, and a small nonlinear autoencoder on the moving
// spot; results in out/autoencoder/.
//   cargo run -p day38 -- autoencoder                      (your code)
//   cargo run -p day38 --features solution -- autoencoder  (the model solution)

use crate::autoencoder::*;
use imgcourse::*;

/// Mean and principal components (largest first) of a set of vectors, with the library's
/// eigenvectors (the PCA unit shows how to compute them).
fn pca(vectors: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let (n, d) = (vectors.len() as f64, vectors[0].len());
    let mut mean = vec![0.0f64; d];
    for v in vectors {
        for i in 0..d {
            mean[i] += v[i] as f64 / n;
        }
    }
    let mut s = vec![vec![0.0f64; d]; d];
    for v in vectors {
        for i in 0..d {
            for j in 0..d {
                s[i][j] += (v[i] as f64 - mean[i]) * (v[j] as f64 - mean[j]) / n;
            }
        }
    }
    let (values, vecs) = linalg::sym_eigen(&s);
    let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    (f(&mean), f(&values), vecs.iter().map(|v| f(v)).collect())
}

/// A scatter plot of two-pixel images: pixel 1 across, pixel 2 up, both from −0.2 to 1.2.
struct Plot(Image<[f32; 3]>);

impl Plot {
    const SIZE: usize = 320;
    fn new() -> Plot {
        Plot(Image::new(Plot::SIZE, Plot::SIZE, [1.0; 3]))
    }
    fn pixel(p: &[f32]) -> (isize, isize) {
        let s = Plot::SIZE as f32;
        (((p[0] + 0.2) / 1.4 * s) as isize, ((1.2 - p[1]) / 1.4 * s) as isize)
    }
    fn dot(&mut self, p: &[f32], r: isize, colour: [f32; 3]) {
        let (cx, cy) = Plot::pixel(p);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r && self.0.contains(cx + dx, cy + dy) {
                    self.0[((cx + dx) as usize, (cy + dy) as usize)] = colour;
                }
            }
        }
    }
    fn line(&mut self, a: &[f32], b: &[f32], colour: [f32; 3]) {
        for k in 0..=200 {
            let t = k as f32 / 200.0;
            self.dot(&[a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], 0, colour);
        }
    }
}

pub fn main() {
    let dir = "out/autoencoder";
    const RED: [f32; 3] = [0.85, 0.2, 0.2];
    const BLUE: [f32; 3] = [0.15, 0.35, 0.85];
    const GREEN: [f32; 3] = [0.1, 0.6, 0.2];
    const GREY: [f32; 3] = [0.35, 0.35, 0.35];

    // 1. Two neighbouring pixels of the micrograph as two-pixel images.
    let cell = samples::cell();
    let mut rng = Rng::new(5);
    let pairs: Vec<Vec<f32>> = (0..300)
        .map(|_| {
            let (x, y) = (rng.below(255), rng.below(256));
            vec![cell[(x, y)], cell[(x + 1, y)]]
        })
        .collect();
    let (mean, values, pcs) = pca(&pairs);
    println!("neighbouring pixels: mean {mean:?}, eigenvalues {values:?}, PC1 {:?}", pcs[0]);
    let keep_pixel1 = reconstruction_error(&pairs, &mean, &[vec![1.0, 0.0]]);
    let keep_pc1 = reconstruction_error(&pairs, &mean, &pcs[..1]);
    let keep_nothing = reconstruction_error(&pairs, &mean, &[]);
    println!("  error keeping pixel 1: {keep_pixel1:.5}, keeping PC1: {keep_pc1:.5}, keeping nothing: {keep_nothing:.5}");

    // 2. PCA of 8 × 8 patches as an autoencoder for the whole micrograph.
    let size = 8;
    let mut patches = Vec::new();
    for y0 in (0..256).step_by(size) {
        for x0 in (0..256).step_by(size) {
            patches.push(cell.crop(x0, y0, size, size).pixels().to_vec());
        }
    }
    let (mean, _, pcs) = pca(&patches);
    for k in [1, 4, 16] {
        let mut out = Image::new(256, 256, 0.0f32);
        for (p, patch) in patches.iter().enumerate() {
            let rebuilt = pca_decode(&pca_encode(patch, &mean, &pcs[..k]), &mean, &pcs[..k]);
            let (x0, y0) = (p % 32 * size, p / 32 * size);
            for (i, v) in rebuilt.iter().enumerate() {
                out[(x0 + i % size, y0 + i / size)] = *v;
            }
        }
        let err = reconstruction_error(&patches, &mean, &pcs[..k]) / (size * size) as f32;
        println!("8 × 8 patches, {k:2} numbers instead of 64: mean squared error per pixel {err:.5}");
        save_gray(&out, format!("{dir}/patches-k{k}.png"));
    }
    save_gray(&cell, format!("{dir}/input.png"));

    // 3. The moving spot: PCA against a trained nonlinear autoencoder.
    let spot = moving_spot(100, 0.02, 7);
    let (mean, _, pcs) = pca(&spot);
    println!("moving spot: PCA error {:.4}", reconstruction_error(&spot, &mean, &pcs[..1]));
    let mut ae = Autoencoder::new(16, 1);
    let mut error = 0.0;
    for step in 0..=20000 {
        error = train_step(&mut ae, &spot, 0.05);
        if step % 5000 == 0 {
            println!("  autoencoder, step {step:4}: error {error:.4}");
        }
    }
    let codes: Vec<f32> = spot.iter().map(|x| forward(&ae, x)[1][0]).collect();
    let (zmin, zmax) = codes.iter().fold((f32::MAX, f32::MIN), |(a, b), &z| (a.min(z), b.max(z)));
    println!("  final error {error:.4}; the codes of the training images run from {zmin:.2} to {zmax:.2}");

    let mut plot = Plot::new();
    plot.line(&[-0.2, 0.0], &[1.2, 0.0], GREY);
    plot.line(&[0.0, -0.2], &[0.0, 1.2], GREY);
    let u = &pcs[0];
    plot.line(&[mean[0] - 1.0 * u[0], mean[1] - 1.0 * u[1]], &[mean[0] + 1.0 * u[0], mean[1] + 1.0 * u[1]], BLUE);
    for x in &spot {
        let rebuilt = pca_decode(&pca_encode(x, &mean, &pcs[..1]), &mean, &pcs[..1]);
        plot.line(x, &rebuilt, RED);
    }
    // the decoder curve: decode codes from zmin to zmax
    let decode = |z: f32| {
        let h = dense(&ae.layers[2], &[z], true);
        dense(&ae.layers[3], &h, false)
    };
    for k in 0..400 {
        let z = zmin + (zmax - zmin) * k as f32 / 399.0;
        plot.dot(&decode(z), 1, GREEN);
    }
    for x in &spot {
        plot.dot(x, 2, [0.0; 3]);
    }
    save_rgb(&plot.0, format!("{dir}/spot.png"));
    println!("wrote the results to {dir}/ (spot.png: black data, blue PC1 line with red errors, green decoder curve)");
}
