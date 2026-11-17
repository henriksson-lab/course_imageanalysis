// Checks for the coding exercises, with small cases and the numbers of the lecture.
// Run: cargo test -p evaluation leakage::

use evaluation::leakage::*;

#[test]
fn c1_knn_small_cases() {
    let train_x = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![1.0, 1.0], vec![0.9, 1.0]];
    let train_y = [false, false, false, true, true];
    assert!(!knn_predict(&train_x, &train_y, &[0.05, 0.05], 3), "a query among the three 'false' points, k = 3: expected false");
    assert!(knn_predict(&train_x, &train_y, &[0.95, 0.9], 1), "a query next to (0.9, 1), k = 1: expected true");
    assert!(knn_predict(&train_x, &train_y, &[0.8, 0.8], 1), "k = 1 copies the nearest label: expected true");
    assert!(!knn_predict(&train_x, &train_y, &[0.8, 0.8], 5), "k = 5: three of the five vote false: expected false");
}

#[test]
fn c2_block_folds() {
    let f = block_folds(12, 8, 4, 3, 1);
    assert!(f.width() == 12 && f.height() == 8, "expected a 12 × 8 fold map, got {} × {}", f.width(), f.height());
    for (x, y) in f.coords() {
        assert!(f[(x, y)] < 3, "pixel ({x}, {y}) has fold {}, but there are only 3 folds", f[(x, y)]);
        let corner = f[(x / 4 * 4, y / 4 * 4)];
        assert_eq!(f[(x, y)], corner, "pixel ({x}, {y}) is in the same 4 × 4 block as ({}, {}) but has another fold", x / 4 * 4, y / 4 * 4);
    }
    let pixels = block_folds(40, 40, 1, 4, 2);
    let used: Vec<usize> = (0..4).map(|k| pixels.pixels().iter().filter(|&&v| v == k).count()).collect();
    assert!(used.iter().all(|&c| c > 300), "random pixels in 4 folds: each fold should get about 400 of 1600 pixels, got {used:?}");
}

#[test]
fn c3_roles_with_buffer() {
    // Fold 1 is the right half of an 8 × 2 image.
    let folds = imgcourse::Image::from_fn(8, 2, |x, _| if x >= 4 { 1 } else { 0 });
    let r = roles(&folds, 1, 2);
    let row: Vec<Role> = (0..8).map(|x| r[(x, 0)]).collect();
    use Role::*;
    assert_eq!(row, vec![Train, Train, Buffer, Buffer, Test, Test, Test, Test], "buffer 2 around the right half");
    let r = roles(&folds, 1, 0);
    assert_eq!(r[(3, 1)], Train, "no buffer: the pixel next to the test half is training");
}

#[test]
fn c4_best_of_one_is_honest() {
    let (reported, fresh) = best_of_k(1, 100, 0.8, 2000, 1);
    assert!((reported - 0.8).abs() < 0.01 && (fresh - 0.8).abs() < 0.01, "one method: reported {reported:.3} and fresh {fresh:.3} should both be about 0.80 (demo question T1)");
}

#[test]
fn c4_best_of_fifty_b2() {
    // Lecture: 50 settings, 100 objects, accuracy 0.8: the best reads 0.89 on average,
    // 0.80 + 2.25 · 0.04.
    let (reported, fresh) = best_of_k(50, 100, 0.8, 1000, 2);
    assert!((reported - 0.89).abs() < 0.01, "50 settings: got the mean reported score {reported:.3}, expected about 0.89");
    assert!((fresh - 0.8).abs() < 0.01, "50 settings: got the mean fresh score {fresh:.3}, expected about 0.80");
}
