// Runs your PCA on the lecture's four images and on 8 × 8 patches of the micrograph;
// results in out/pca/.
//   cargo run -p pca                      (your code)
//   cargo run -p pca --features solution  (the model solution)

use imgcourse::*;
use pca::*;

fn main() {
    let dir = "out/pca";
    // The lecture's four two-pixel images.
    let four = [[0.2, 0.4], [0.4, 0.2], [0.6, 0.8], [0.8, 0.6]];
    let (mean, s) = mean_and_covariance(&four);
    let (lambda, u) = eigen2(s);
    println!("four images: mean {mean:?}, S = {s:?}");
    println!("eigenvalues {lambda:?}, PC1 {:?}, PC2 {:?}", u[0], u[1]);

    // The first 8 principal components of 8 × 8 patches of the micrograph. After each one,
    // remove it from S ("deflation") so that power iteration finds the next one.
    let size = 8;
    let (_, mut s) = covariance(&patches(&samples::cell(), size, 2));
    let total: f32 = (0..s.len()).map(|i| s[i][i]).sum();
    let mut sheet = Image::new(8 * (size + 1) - 1, size, 0.5f32);
    for k in 0..8 {
        let (lambda, pc) = power_iteration(&s, 300);
        println!("PC{}: {:.1}% of the variance", k + 1, 100.0 * lambda / total);
        let m = pc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (i, v) in pc.iter().enumerate() {
            sheet[(k * (size + 1) + i % size, i / size)] = 0.5 + 0.5 * v / m;
        }
        for i in 0..s.len() {
            for j in 0..s.len() {
                s[i][j] -= lambda * pc[i] * pc[j];
            }
        }
    }
    // Enlarge 8 times so that the pixels can be seen.
    let big = Image::from_fn(sheet.width() * 8, sheet.height() * 8, |x, y| sheet[(x / 8, y / 8)]);
    save_gray(&big, format!("{dir}/patch-pcs.png"));
    println!("wrote {dir}/patch-pcs.png (PC1 to PC8 of 8 × 8 patches, 0 = grey)");
}
