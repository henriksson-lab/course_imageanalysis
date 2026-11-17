// Coding exercises for "Contrastive learning". Replace each `todo!()` with your code,
// then check it with `cargo test -p selfsup contrastive::` and look at `cargo run -p selfsup -- contrastive`.
#![allow(unused_variables, unused_imports)]

/// C1. The dot product of two vectors (for unit vectors: the cosine similarity).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    todo!("C1: dot")
}

/// C2. The InfoNCE loss, one direction: for every anchor `a[i]`, a softmax over the scores
/// `a[i] · b[j] / tau` of all `j`, and minus the log of the probability of its partner `b[i]`;
/// the mean over `i`. Also returns the probabilities `p[i][j]`.
///
/// Hint: for each `i`, compute the scores, subtract the largest before `exp` (as in the
/// attention unit), divide by the sum: that row is `p[i]`; add `-p[i][i].ln()` to the loss.
pub fn info_nce(a: &[Vec<f32>], b: &[Vec<f32>], tau: f32) -> (f32, Vec<Vec<f32>>) {
    todo!("C2: info_nce")
}

/// C3. The gradient of the one-direction loss with respect to every `a[i]` and every `b[j]`:
/// with `g[i][j] = (p[i][j] − [i = j]) / (n tau)`,
/// `dL/da[i] = Σ_j g[i][j] b[j]` and `dL/db[j] = Σ_i g[i][j] a[i]`.
///
/// Hint: get `p` from `info_nce`, then one double loop over `i` and `j` adds to both.
pub fn info_nce_grad(a: &[Vec<f32>], b: &[Vec<f32>], tau: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    todo!("C3: info_nce_grad")
}

/// C4. Back through the normalisation `z = u / |u|`: given `dL/dz`, return `dL/du`
/// `= (dL/dz − z (z · dL/dz)) / |u|` (the part along `z` is removed: it would only change the length).
pub fn unit_backward(u: &[f32], dz: &[f32]) -> Vec<f32> {
    todo!("C4: unit_backward")
}
