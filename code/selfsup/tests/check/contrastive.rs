// Checks for the coding exercises of "Contrastive learning".
// Run: cargo test -p selfsup contrastive::

use selfsup::contrastive::*;

fn unit_at(cos: f32) -> Vec<f32> {
    vec![cos, (1.0 - cos * cos).max(0.0).sqrt()]
}

#[test]
fn c1_dot() {
    assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]), 12.0, "1·4 + 2·(−5) + 3·6 = 12");
}

#[test]
fn c2_info_nce_b2() {
    // B2: one anchor (1, 0) whose cosines with b1, b2, b3 are 0.9, 0.1, −0.5; only row 0 matters,
    // so give the other anchors the same vector and compare the first row's terms by hand.
    let a = [vec![1.0, 0.0]];
    let b = vec![unit_at(0.9), unit_at(0.1), unit_at(-0.5)];
    // a square batch is needed: pad the anchors with copies of b2 and b3 and read p[0]
    let a3 = vec![a[0].clone(), b[1].clone(), b[2].clone()];
    let (_, p) = info_nce(&a3, &b, 0.5);
    assert!((p[0][0] - 0.792).abs() < 1e-3, "B2a: probability of the partner at tau 0.5: expected 0.792, got {}", p[0][0]);
    assert!((-p[0][0].ln() - 0.233).abs() < 1e-3, "B2a: loss term 0.233");
    let (_, p) = info_nce(&a3, &b, 0.1);
    assert!((-p[0][0].ln() - 0.00034).abs() < 2e-5, "B2b: at tau 0.1 the loss term is 0.00034, got {}", -p[0][0].ln());
    let same = vec![vec![1.0, 0.0]; 3];
    let (l, _) = info_nce(&same, &same, 0.2);
    assert!((l - 3f32.ln()).abs() < 1e-5, "B3: a collapsed batch of 3 costs ln 3 = 1.099, got {l}");
}

#[test]
fn c3_gradient_matches_finite_differences() {
    let a = vec![unit_at(0.3), unit_at(-0.8), vec![-0.6, 0.8], vec![0.0, -1.0]];
    let b = vec![unit_at(0.5), vec![-1.0, 0.0], unit_at(0.9), vec![0.6, -0.8]];
    let tau = 0.3;
    let (ga, gb) = info_nce_grad(&a, &b, tau);
    let h = 1e-3;
    for i in 0..4 {
        for k in 0..2 {
            let (mut ap, mut am) = (a.clone(), a.clone());
            ap[i][k] += h;
            am[i][k] -= h;
            let num = (info_nce(&ap, &b, tau).0 - info_nce(&am, &b, tau).0) / (2.0 * h);
            assert!((num - ga[i][k]).abs() < 2e-3, "dL/da[{i}][{k}]: formula {}, finite difference {num}", ga[i][k]);
            let (mut bp, mut bm) = (b.clone(), b.clone());
            bp[i][k] += h;
            bm[i][k] -= h;
            let num = (info_nce(&a, &bp, tau).0 - info_nce(&a, &bm, tau).0) / (2.0 * h);
            assert!((num - gb[i][k]).abs() < 2e-3, "dL/db[{i}][{k}]: formula {}, finite difference {num}", gb[i][k]);
        }
    }
}

#[test]
fn c4_unit_backward() {
    // u = (3, 4), |u| = 5, z = (0.6, 0.8). A gradient along z has no effect on z: 0.
    let g = unit_backward(&[3.0, 4.0], &[0.6, 0.8]);
    assert!(g.iter().all(|v| v.abs() < 1e-6), "a gradient along z gives 0, got {g:?}");
    // across z: (−0.8, 0.6) / 5
    let g = unit_backward(&[3.0, 4.0], &[-0.8, 0.6]);
    assert!((g[0] + 0.16).abs() < 1e-6 && (g[1] - 0.12).abs() < 1e-6, "across z: expected (−0.16, 0.12), got {g:?}");
}
