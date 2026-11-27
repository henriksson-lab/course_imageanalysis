// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p shapematch fft_docking::

use imgcourse::fft::fft2;
use imgcourse::{Complex, Image};
use shapematch::fft_docking::*;

fn assert_all_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: {} values, expected {}", got.len(), expected.len());
    for (t, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() < 1e-3, "{what}: got {got:?}, expected {expected:?} (first difference at index {t})");
    }
}

#[test]
fn c1_score_1d_b1() {
    let rho = -9.0;
    let a = [rho, rho, rho, 1., 0., 0., 0., 0.];
    let b = [1., 1., 0., 0., 0., 0., 0., 0.];
    // S(1) = 2ρ, S(2) = ρ + 1, S(3) = 1, S(4) = 0; S(7) wraps around onto cell 0
    assert_all_close(&score_1d(&a, &b), &[-18., -18., -8., 1., 0., 0., 0., -9.], "B1");
}

#[test]
fn c2_receptor_grid_b1() {
    let mask = Image::from_rows(&[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    let a = receptor_grid(&mask, 1, -9.0);
    assert_all_close(a.row(0), &[-9., -9., -9., 1., 0., 0., 0., 0.], "B1 receptor, surface 1 cell thick");
    let a2 = receptor_grid(&mask, 2, -9.0);
    assert_all_close(a2.row(0), &[-9., -9., -9., 1., 1., 0., 0., 0.], "surface 2 cells thick");
}

#[test]
fn c2_surface_is_a_disk_around_the_core() {
    // one core cell in the middle of a 7 × 7 grid, thickness 2: the 12 cells with dx² + dy² <= 4
    let mask = Image::from_fn(7, 7, |x, y| if (x, y) == (3, 3) { 1.0 } else { 0.0 });
    let a = receptor_grid(&mask, 2, -5.0);
    let surface = a.pixels().iter().filter(|&&v| v == 1.0).count();
    assert_eq!(surface, 12, "a disk of radius 2 has 13 cells, minus the core: 12 surface cells, got {surface}");
    assert_eq!(a[(3, 3)], -5.0, "the core cell holds ρ");
    assert_eq!(a[(4, 4)], 1.0, "(4, 4) is at distance √2 ≤ 2: surface");
    assert_eq!(a[(5, 5)], 0.0, "(5, 5) is at distance √8 > 2: outside the layer");
}

#[test]
fn c3_fft_map_equals_the_direct_score() {
    let m = 16;
    let receptor = Image::from_fn(m, m, |x, y| ((x * 7 + y * 3) % 5) as f32 - 2.0);
    let ligand = Image::from_fn(5, 5, |x, y| if (x + y) % 3 == 0 { 1.0 } else { 0.0 });
    let fa = fft2(&receptor.map(Complex::from), false);
    let map = score_map(&fa, &ligand);
    for (x, y) in [(0, 0), (3, 4), (15, 1), (8, 15)] {
        let d = score_at(&receptor, &ligand, x, y);
        assert!((map[(x, y)] - d).abs() < 1e-3, "shift ({x}, {y}): FFT map {}, direct {d}", map[(x, y)]);
    }
    let b = ligand_grid(&ligand, m);
    assert_eq!(b[(0, 0)], ligand[(2, 2)], "the ligand's centre cell goes to (0, 0)");
    assert_eq!(b[(m - 2, m - 2)], ligand[(0, 0)], "cell (0, 0) of the ligand wraps to (m − 2, m − 2)");
}

#[test]
fn c4_rotate_ligand_quarter_turn() {
    // a cell one to the right of the centre goes below it after 90° clockwise on screen
    let lig = Image::from_fn(5, 5, |x, y| if (x, y) == (3, 2) { 1.0 } else { 0.0 });
    let r = rotate_ligand(&lig, 90.0);
    assert_eq!(r[(2, 3)], 1.0, "90° clockwise: (3, 2) → (2, 3)");
    assert_eq!(r.sum(), 1.0, "one cell stays one cell");
    let back = rotate_ligand(&rotate_ligand(&lig, 30.0), -30.0);
    assert_eq!(back[(3, 2)], 1.0, "turning by 30° and back returns the cell");
}

#[test]
fn c5_dock_finds_a_planted_pose() {
    // An L-shaped ligand and a receptor with an L-shaped pocket that fits it turned by 90°.
    let lig = Image::from_rows(&[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]);
    let turned = rotate_ligand(&lig, 90.0);
    let (px, py) = (10, 7); // ligand centre at the true pose: the L lies inside the block
    let mask = Image::from_fn(32, 32, |x, y| {
        let in_block = (4..=16).contains(&x) && (6..=14).contains(&y);
        let (i, j) = (x as isize - px as isize + 2, y as isize - py as isize + 2);
        let in_pocket = turned.contains(i, j) && turned[(i as usize, j as usize)] > 0.0;
        if in_block && !in_pocket { 1.0 } else { 0.0 }
    });
    let receptor = receptor_grid(&mask, 1, -9.0);
    let poses = dock(&receptor, &lig, 90.0);
    assert_eq!(poses.len(), 4, "angles 0, 90, 180, 270");
    let best = poses.iter().copied().max_by(|a, b| a.score.total_cmp(&b.score)).unwrap();
    assert_eq!((best.angle, best.x, best.y), (90.0, px, py), "best pose {best:?}, expected 90° at ({px}, {py})");
    assert!(best.score > 0.0, "the true pose scores contacts only");
}
