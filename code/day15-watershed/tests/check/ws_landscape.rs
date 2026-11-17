// Checks for the coding exercises, using the signals of the pen-and-paper exercises.
// Run: cargo test -p day15 ws_landscape::
// Basins A, B, C, D of the exercise are labels 1, 2, 3, 4 here.

use day15::ws_landscape::*;

const B1: [f32; 12] = [7., 1., 4., 3., 4., 6., 5., 1., 6., 4., 2., 3.];
const STEP: [f32; 10] = [1., 1., 1., 1., 5., 5., 5., 2., 2., 2.];

#[test]
fn c1_minima_b1() {
    assert_eq!(regional_minima(&B1), [0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0], "B1 (a): minima at samples 1, 3, 7, 10");
}

#[test]
fn c1_minima_plateaus_b4_b5() {
    assert_eq!(regional_minima(&STEP), [1, 1, 1, 1, 0, 0, 0, 2, 2, 2], "B4 (b): the step has two minimum plateaus");
    let d = abs_derivative(&STEP);
    assert_eq!(d, [0., 0., 0., 2., 2., 0., 1.5, 1.5, 0., 0.], "B4 (a): |f'| (this is the given helper)");
    assert_eq!(regional_minima(&d), [1, 1, 1, 0, 0, 2, 0, 0, 3, 3], "B4 (b): |f'| has three minima");
    let noisy = [3., 2., 3., 2., 3., 2., 3.];
    assert_eq!(regional_minima(&noisy), [0, 1, 0, 2, 0, 3, 0], "B5 (a): three minima");
    assert_eq!(regional_minima(&[5., 5., 5.]), [1, 1, 1], "a flat signal is one minimum");
}

#[test]
fn c2_drop_b6() {
    assert_eq!(drop_of_water(&B1, 5), 3, "B6: a drop on sample 5 ends at sample 3");
    assert_eq!(drop_of_water(&B1, 8), 7, "B6: a drop on sample 8 ends at sample 7");
    assert_eq!(drop_of_water(&B1, 10), 10, "a drop on a minimum stays there");
}

#[test]
fn c3_flood_b1() {
    let got = flood(&B1, &regional_minima(&B1));
    assert_eq!(got, [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], "B1 (c): A A A B B B C C C D D D");
}

#[test]
fn c3_flood_hminima_b2() {
    // B2 (b): flood f from the minima of HMIN_2(f): samples 1, 7 and 9-11.
    let sources = [0, 1, 0, 0, 0, 0, 0, 2, 0, 3, 3, 3];
    let got = flood(&B1, &sources);
    assert_eq!(got, [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3], "B2 (b): A A A A A A B B B C C C");
}

#[test]
fn c3_flood_markers_b3() {
    let mut markers = [0; 12];
    markers[3] = 1;
    markers[10] = 2;
    let got = flood(&B1, &markers);
    assert_eq!(got, [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], "B3 (b): A × 8, then B × 4: the minima at 1 and 7 are drowned");
}

#[test]
fn c3_flood_step_b4() {
    let f = flood(&STEP, &regional_minima(&STEP));
    assert_eq!(f, [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], "B4 (b): flooding f gives two regions");
    let d = abs_derivative(&STEP);
    let g = flood(&d, &regional_minima(&d));
    assert_eq!(g, [1, 1, 1, 1, 2, 2, 2, 3, 3, 3], "B4 (b): flooding |f'| gives three regions, boundaries on the edges");
}

#[test]
fn c4_dams_b1_b3() {
    let labels = flood(&B1, &regional_minima(&B1));
    assert_eq!(dams(&B1, &labels), [(2, 4.), (5, 6.), (8, 6.)], "B1 (c): dams on samples 2, 5, 8, heights 4, 6, 6");
    assert_eq!(dams(&B1, &[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]), [(8, 6.)], "B3 (b): one dam on sample 8, height 6");
}

#[test]
fn c5_hminima_b2() {
    assert_eq!(h_minima(&B1, 2.0), [7., 3., 4., 4., 4., 6., 5., 3., 6., 4., 4., 4.], "B2 (a): HMIN_2");
    assert_eq!(h_minima(&B1, 1.0), [7., 2., 4., 4., 4., 6., 5., 2., 6., 4., 3., 3.], "B2 (c): HMIN_1");
    let survivors = regional_minima(&h_minima(&B1, 2.0)).iter().copied().max().unwrap();
    assert_eq!(survivors, 3, "B2 (a): three minima survive h = 2");
}

#[test]
fn c5_hminima_b5() {
    let noisy = [3., 2., 3., 2., 3., 2., 3.];
    assert_eq!(h_minima(&noisy, 1.0), [3.; 7], "B5 (b): h = 1 fills all three basins to 3");
    let left = regional_minima(&h_minima(&noisy, 0.5)).iter().copied().max().unwrap();
    assert_eq!(left, 3, "B5 (b): with h = 0.5 all three survive");
}
