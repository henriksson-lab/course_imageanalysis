#![allow(clippy::needless_range_loop)] // index loops read like the math
//! Small dense linear algebra in `f64`, for the days after the linear algebra has been taught:
//! solving a linear system, least squares, and the eigenvectors of a symmetric matrix.
//! Matrices are `Vec<Vec<f64>>`, row by row: `a[row][column]`.

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
/// Returns `None` if `a` is (numerically) singular.
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    assert!(a.len() == n && a.iter().all(|r| r.len() == n), "solve: a must be n × n and b of length n");
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &bi)| r.iter().copied().chain([bi]).collect()).collect();
    let scale = a.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs())).max(f64::MIN_POSITIVE);
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col].abs() <= 1e-12 * scale {
            return None;
        }
        m.swap(col, pivot);
        for row in col + 1..n {
            let f = m[row][col] / m[col][col];
            for k in col..=n {
                m[row][k] -= f * m[col][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| m[row][k] * x[k]).sum();
        x[row] = (m[row][n] - s) / m[row][row];
    }
    Some(x)
}

/// The least-squares solution of `a x ≈ b` for a tall `a` (more rows than columns), via the
/// normal equations `aᵀa x = aᵀb`. Returns `None` if the columns are linearly dependent.
pub fn least_squares(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.first().map_or(0, |r| r.len());
    let ata: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| a.iter().map(|r| r[i] * r[j]).sum()).collect()).collect();
    let atb: Vec<f64> = (0..n).map(|i| a.iter().zip(b).map(|(r, &bk)| r[i] * bk).sum()).collect();
    solve(&ata, &atb)
}

/// The matrix product `a b`.
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, m, p) = (a.len(), b.len(), b.first().map_or(0, |r| r.len()));
    assert!(a.iter().all(|r| r.len() == m), "matmul: sizes do not match");
    (0..n).map(|i| (0..p).map(|j| (0..m).map(|k| a[i][k] * b[k][j]).sum()).collect()).collect()
}

/// The transpose.
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.first().map_or(0, |r| r.len());
    (0..m).map(|j| a.iter().map(|r| r[j]).collect()).collect()
}

/// Eigenvalues and unit eigenvectors of a symmetric matrix (Jacobi rotations), sorted with
/// the largest eigenvalue first. Returns `(values, vectors)` with `vectors[k]` belonging to
/// `values[k]`.
pub fn sym_eigen(s: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = s.len();
    let mut a: Vec<Vec<f64>> = s.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - sn * akq;
                    a[k][q] = sn * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - sn * aqk;
                    a[q][k] = sn * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - sn * vq;
                    row[q] = sn * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = order.iter().map(|&i| a[i][i]).collect();
    let vectors = order.iter().map(|&i| v.iter().map(|row| row[i]).collect()).collect();
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_and_least_squares() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let x = solve(&a, &[3.0, 5.0]).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]).is_none());
        // fit y = 1 + 2t exactly
        let rows: Vec<Vec<f64>> = (0..5).map(|t| vec![1.0, t as f64]).collect();
        let ys: Vec<f64> = (0..5).map(|t| 1.0 + 2.0 * t as f64).collect();
        let c = least_squares(&rows, &ys).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-9 && (c[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn eigen() {
        let (l, u) = sym_eigen(&[vec![0.05, 0.03], vec![0.03, 0.05]]);
        assert!((l[0] - 0.08).abs() < 1e-12 && (l[1] - 0.02).abs() < 1e-12);
        assert!((u[0][0].abs() - 0.5f64.sqrt()).abs() < 1e-9 && u[0][0] * u[0][1] > 0.0);
        let s = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 1.0]];
        let (l, u) = sym_eigen(&s);
        for k in 0..3 {
            let su: Vec<f64> = (0..3).map(|i| (0..3).map(|j| s[i][j] * u[k][j]).sum()).collect();
            for i in 0..3 {
                assert!((su[i] - l[k] * u[k][i]).abs() < 1e-9);
            }
        }
        assert!(l[0] >= l[1] && l[1] >= l[2]);
    }
}
