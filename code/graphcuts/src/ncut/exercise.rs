// Coding exercises for "Spectral partitioning and normalized cuts". Replace each `todo!()`
// with your code, then check it with `cargo test -p graphcuts ncut::` and look at `cargo run -p graphcuts -- ncut`.
#![allow(unused_variables, unused_imports)]

// A graph with n nodes is its symmetric matrix of affinities `w[p][q]` (0 on the diagonal).

/// C1. Affinities between points: w = exp(−d² / 2σ²) for every pair at distance d, and 0 on
/// the diagonal (no node is joined to itself).
pub fn affinities(points: &[[f64; 2]], sigma: f64) -> Vec<Vec<f64>> {
    todo!("C1: affinities")
}

/// C2a. The degrees d_p = Σ_q w_pq and the graph Laplacian L = D − W.
///
/// Hint: `row.iter().sum()` adds up one row of `w`.
pub fn laplacian(w: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    todo!("C2a: laplacian")
}

/// C2b. The quadratic form xᵀ L x.
pub fn quadratic_form(l: &[Vec<f64>], x: &[f64]) -> f64 {
    todo!("C2b: quadratic_form")
}

/// C3. The cut between A (`in_a[p]` true) and B (the rest), and the normalized cut
/// Ncut = cut / vol(A) + cut / vol(B), with vol = the sum of the degrees.
///
/// Hint: one double loop over all pairs (p, q): w[p][q] counts towards vol(A) if p is in A,
/// towards vol(B) otherwise, and towards the cut if p is in A and q is not.
pub fn ncut(w: &[Vec<f64>], in_a: &[bool]) -> (f64, f64) {
    todo!("C3: ncut")
}

/// C4. The normalized Fiedler vector: the eigenvector x of L x = μ D x for the second-smallest
/// μ (math slide "A generalized eigenproblem"), by power iteration without solving any
/// equation. With y = D^{1/2} x the problem is N y = μ y, N = I − D^{−1/2} W D^{−1/2}. The
/// matrix M = 2I − N = I + D^{−1/2} W D^{−1/2} has the eigenvalues 2 − μ, so its largest are
/// 2 (the trivial y₁ = D^{1/2} 1) and 2 − μ₂. Power iteration on M, keeping y orthogonal to
/// y₁, finds y₂. Returns μ₂ and x = D^{−1/2} y₂ (scaled so that its largest entry is ±1).
///
/// Hint: with s = (√d₁, ..., √d_n) scaled to length 1, "keep y orthogonal to y₁" is
/// y ← y − (y · s) s. Each step: z = y + D^{−1/2} W D^{−1/2} y, take out the part along s,
/// the eigenvalue of M is z · y (y has length 1), then scale z to length 1. Start from any
/// vector, e.g. 1, 2, 3, ... (with the part along s taken out).
pub fn fiedler(w: &[Vec<f64>], iterations: usize) -> (f64, Vec<f64>) {
    todo!("C4: fiedler")
}

/// C5. The sweep (slide "From eigenvector to segmentation"): sort the nodes by `x`, try
/// every split "the k smallest against the rest" (k = 1, ..., n − 1) and keep the one with
/// the smallest Ncut. Returns that split (`true` = in A, the nodes with the small values)
/// and its Ncut.
///
/// Hint: `order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap())` sorts the node numbers
/// by their values. Then add the nodes to A one at a time, and compute `ncut` after each.
pub fn sweep(w: &[Vec<f64>], x: &[f64]) -> (Vec<bool>, f64) {
    todo!("C5: sweep")
}
