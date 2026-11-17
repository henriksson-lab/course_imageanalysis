// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p gpu-patterns

use gpu_patterns::*;

#[test]
fn c1_reduce_tree_b1() {
    let rows = reduce_tree(&[5.0, 2.0, 7.0, 1.0, 3.0, 8.0, 4.0, 6.0]);
    let expected: Vec<Vec<f32>> = vec![vec![7.0, 8.0, 11.0, 10.0], vec![15.0, 21.0], vec![36.0]];
    assert_eq!(rows, expected, "B1: the rows after each step");
    let odd = reduce_tree(&[1.0, 2.0, 3.0]);
    assert_eq!(odd.last(), Some(&vec![6.0]), "three values: the odd one is carried over; got {odd:?}");
}

#[test]
fn c1_order_of_additions() {
    // Floats are not associative: 10^8 followed by fifteen ones, in float32 (floats near 10^8
    // are 8 apart). Added one by one, every 1 is lost; the tree adds the ones in pairs first
    // and keeps 8 of the 15.
    let mut v = vec![1.0f32; 16];
    v[0] = 1e8;
    let tree = reduce_tree(&v).last().unwrap()[0];
    let looped = v.iter().fold(0.0f32, |s, &x| s + x);
    assert_eq!(looped, 1e8, "the loop loses every 1");
    assert_eq!(tree, 1e8 + 8.0, "the tree: got {tree}, expected 10^8 + 8");
}

#[test]
fn c2_hillis_steele_b2a() {
    let (scan, adds) = scan_hillis_steele(&[3.0, 1.0, 7.0, 0.0, 4.0, 1.0, 6.0, 3.0]);
    assert_eq!(scan, vec![3.0, 4.0, 11.0, 11.0, 15.0, 16.0, 22.0, 25.0], "B2 (a): the running totals");
    assert_eq!(adds, 17, "B2 (a): 7 + 6 + 4 = 17 additions");
    let (scan, adds) = scan_hillis_steele(&[1.0; 16]);
    assert_eq!((scan[15], adds), (16.0, 49), "16 ones: last total 16, 16 · 4 − 15 = 49 additions");
}

#[test]
fn c3_blelloch_b2b() {
    let (scan, adds) = scan_blelloch(&[3.0, 1.0, 7.0, 0.0, 4.0, 1.0, 6.0, 3.0]);
    assert_eq!(scan, vec![0.0, 3.0, 4.0, 11.0, 11.0, 15.0, 16.0, 22.0], "B2 (b): the exclusive scan");
    assert_eq!(adds, 14, "B2 (b): 2 (n − 1) = 14 additions");
    let x: Vec<f32> = (0..64).map(|i| ((i * 7) % 5) as f32).collect();
    let (scan, adds) = scan_blelloch(&x);
    let mut total = 0.0;
    for i in 0..64 {
        assert_eq!(scan[i], total, "64 values, element {i}: got {}, expected {total}", scan[i]);
        total += x[i];
    }
    assert_eq!(adds, 126, "64 values: 2 · 63 additions");
}

#[test]
fn c4_compaction_b3() {
    let values: Vec<char> = "abcdefghijkl".chars().collect();
    let keep = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1].map(|b| b == 1);
    let out: String = compact(&values, &keep).into_iter().collect();
    assert_eq!(out, "acdgijl", "B3: the kept elements, in order");
    assert!(compact(&values, &[false; 12]).is_empty(), "nothing kept: an empty list");
}

#[test]
fn c5_histogram_b5() {
    let values = [0, 1, 1, 2, 0, 0, 0, 7, 3, 0, 0, 1, 0, 5, 0, 0, 2, 0, 0, 0, 6, 0, 1, 0, 0, 0, 4, 0, 0, 2, 0, 0];
    let h = atomic_histogram(&values, 8, 8);
    assert_eq!(h.counts, vec![20, 4, 3, 1, 1, 1, 1, 1], "B5 (a): the histogram");
    assert_eq!(h.global_cycles, 20, "B5 (a): bin 0 gets 20 updates in one queue");
    assert_eq!(h.private_cycles, 10, "B5 (b): 6 for the busiest private bin, + 4 to add the 4 block histograms");
}
