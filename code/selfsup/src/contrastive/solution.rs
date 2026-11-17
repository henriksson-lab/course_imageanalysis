// Model solution for the coding exercises of "Contrastive learning".

/// C1. The dot product of two vectors (for unit vectors: the cosine similarity).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// C2. The InfoNCE loss, one direction: for every anchor `a[i]`, a softmax over the scores
/// `a[i] · b[j] / tau` of all `j`, and minus the log of the probability of its partner `b[i]`;
/// the mean over `i`. Also returns the probabilities `p[i][j]`.
pub fn info_nce(a: &[Vec<f32>], b: &[Vec<f32>], tau: f32) -> (f32, Vec<Vec<f32>>) {
    let n = a.len();
    let mut loss = 0.0;
    let mut p = Vec::with_capacity(n);
    for i in 0..n {
        let s: Vec<f32> = (0..n).map(|j| dot(&a[i], &b[j]) / tau).collect();
        let m = s.iter().cloned().fold(f32::MIN, f32::max); // subtract the largest score
        let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
        let total: f32 = e.iter().sum();
        loss -= (e[i] / total).ln();
        p.push(e.iter().map(|v| v / total).collect());
    }
    (loss / n as f32, p)
}

/// C3. The gradient of the one-direction loss with respect to every `a[i]` and every `b[j]`:
/// with `g[i][j] = (p[i][j] − [i = j]) / (n tau)`,
/// `dL/da[i] = Σ_j g[i][j] b[j]` and `dL/db[j] = Σ_i g[i][j] a[i]`.
pub fn info_nce_grad(a: &[Vec<f32>], b: &[Vec<f32>], tau: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = a.len();
    let d = a[0].len();
    let (_, p) = info_nce(a, b, tau);
    let mut ga = vec![vec![0.0; d]; n];
    let mut gb = vec![vec![0.0; d]; n];
    for i in 0..n {
        for j in 0..n {
            let g = (p[i][j] - if i == j { 1.0 } else { 0.0 }) / (n as f32 * tau);
            for k in 0..d {
                ga[i][k] += g * b[j][k];
                gb[j][k] += g * a[i][k];
            }
        }
    }
    (ga, gb)
}

/// C4. Back through the normalisation `z = u / |u|`: given `dL/dz`, return `dL/du`
/// `= (dL/dz − z (z · dL/dz)) / |u|` (the part along `z` is removed: it would only change the length).
pub fn unit_backward(u: &[f32], dz: &[f32]) -> Vec<f32> {
    let len = dot(u, u).sqrt().max(1e-12);
    let z: Vec<f32> = u.iter().map(|v| v / len).collect();
    let along = dot(&z, dz);
    dz.iter().zip(&z).map(|(g, zk)| (g - zk * along) / len).collect()
}
