// Model solution for the coding exercises of "Fine-tuning strategies".

pub use crate::common::*;

/// C1. The penalty λ Σ (w − w₀)² that keeps the weights near the pretrained ones ("L2-SP"):
/// add its gradient 2λ (w − w₀) to the gradient `g` of every weight of the backbone: all layers
/// but the last (the new head has no pretrained weights to stay near), and not the biases.
pub fn add_pull_to_start(g: &mut Net, net: &Net, start: &Net, lambda: f32) {
    for l in 0..net.len() - 1 {
        for o in 0..net[l].w.len() {
            for i in 0..net[l].w[o].len() {
                g[l].w[o][i] += 2.0 * lambda * (net[l].w[o][i] - start[l].w[o][i]);
            }
        }
    }
}

/// C2. Numbers to store for a change of an m × n weight matrix: the full change (m·n), and a
/// low-rank change B·A with B m × r and A r × n.
pub fn param_counts(m: usize, n: usize, r: usize) -> (usize, usize) {
    (m * n, r * (m + n))
}

/// The matrix product of an a × b and a b × c matrix (lists of rows).
pub fn matmul(x: &[Vec<f32>], y: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter().map(|row| (0..y[0].len()).map(|j| row.iter().zip(y).map(|(v, yr)| v * yr[j]).sum()).collect()).collect()
}

/// C3. Fit a rank-r change B·A to a target change Δ (m × n) by gradient descent on
/// ‖B·A − Δ‖² (sum of squares). Start as LoRA does: A small and random (0.1 · normal), B = 0.
/// Gradients: with E = B·A − Δ, ∂/∂B = 2 E Aᵀ and ∂/∂A = 2 Bᵀ E. Returns (B, A).
pub fn fit_low_rank(delta: &[Vec<f32>], r: usize, steps: usize, lr: f32, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (m, n) = (delta.len(), delta[0].len());
    let mut rng = imgcourse::Rng::new(seed);
    let mut a: Vec<Vec<f32>> = (0..r).map(|_| (0..n).map(|_| 0.1 * rng.normal()).collect()).collect();
    let mut b = vec![vec![0.0f32; r]; m];
    for _ in 0..steps {
        let ba = matmul(&b, &a);
        let e: Vec<Vec<f32>> = (0..m).map(|i| (0..n).map(|j| ba[i][j] - delta[i][j]).collect()).collect();
        // gradients
        let mut gb = vec![vec![0.0f32; r]; m];
        let mut ga = vec![vec![0.0f32; n]; r];
        for i in 0..m {
            for k in 0..r {
                for j in 0..n {
                    gb[i][k] += 2.0 * e[i][j] * a[k][j];
                    ga[k][j] += 2.0 * b[i][k] * e[i][j];
                }
            }
        }
        for i in 0..m {
            for k in 0..r {
                b[i][k] -= lr * gb[i][k];
            }
        }
        for k in 0..r {
            for j in 0..n {
                a[k][j] -= lr * ga[k][j];
            }
        }
    }
    (b, a)
}

/// C4. The relative error ‖B·A − Δ‖ / ‖Δ‖ (square roots of sums of squares).
pub fn rel_error(delta: &[Vec<f32>], b: &[Vec<f32>], a: &[Vec<f32>]) -> f32 {
    let ba = matmul(b, a);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..delta.len() {
        for j in 0..delta[0].len() {
            num += (ba[i][j] - delta[i][j]).powi(2);
            den += delta[i][j].powi(2);
        }
    }
    (num / den).sqrt()
}
