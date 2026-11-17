// Checks for the coding exercises, using the grids of the pen-and-paper exercises.
// Run: cargo test -p watershed ws_flooding::
// The exercise writes pixels as (row, column); the code uses (x, y) = (column, row).
// Markers A, B are labels 1, 2; a watershed line pixel is 0.

use imgcourse::{Image, Rng};
use watershed::ws_flooding::*;

fn b1() -> (Image<f32>, Image<u32>) {
    let f = Image::from_rows(&[[3., 2., 3., 5., 6.], [2., 1., 2., 4., 5.], [3., 2., 4., 3., 2.], [5., 4., 3., 2., 1.], [6., 5., 3., 1., 0.]]);
    let mut m = Image::new(5, 5, 0u32);
    m[(1, 1)] = 1; // A at row 1, column 1
    m[(4, 4)] = 2; // B at row 4, column 4
    (f, m)
}

fn line_pixels(labels: &Image<u32>) -> Vec<(usize, usize)> {
    // as (row, column), to compare with the exercise
    labels.coords().filter(|&p| labels[p] == 0).map(|(x, y)| (y, x)).collect()
}

/// A 1D signal as an image one row high, with markers at the given samples.
fn signal(f: &[f32], markers: &[(usize, u32)]) -> (Image<f32>, Image<u32>) {
    let mut m = Image::new(f.len(), 1, 0u32);
    for &(i, l) in markers {
        m[(i, 0)] = l;
    }
    (Image::from_vec(f.len(), 1, f.to_vec()), m)
}

#[test]
fn c1_minima_b1() {
    let (f, _) = b1();
    let (minima, n) = regional_minima(&f, false);
    assert_eq!(n, 2, "the B1 grid has 2 regional minima (the two markers)");
    assert_eq!((minima[(1, 1)], minima[(4, 4)]), (1, 2), "minima at row 1, column 1 and row 4, column 4");
}

#[test]
fn c1_minima_plateau_and_connectivity() {
    // A plateau of three 1s is one minimum; a pixel whose only lower neighbour is diagonal is a
    // minimum with 4 neighbours but not with 8.
    let f = Image::from_rows(&[[5., 5., 5., 5., 5.], [5., 1., 1., 1., 5.], [5., 5., 5., 5., 3.], [5., 5., 5., 2., 5.]]);
    let (_, n4) = regional_minima(&f, false);
    let (m8, n8) = regional_minima(&f, true);
    assert_eq!(n8, 2, "8 neighbours: the plateau of 1s and the 2; got {n8}");
    assert_eq!(n4, 3, "4 neighbours: also the 3, whose lower neighbour 2 is only diagonal; got {n4}");
    assert!(m8[(1, 1)] == m8[(3, 1)] && m8[(1, 1)] > 0, "the plateau of 1s is one minimum");
}

#[test]
fn c2_first_pops_b1() {
    let (f, m) = b1();
    let (_, order) = watershed(&f, &m, false, true, &mut HeapQueue::new());
    let rc: Vec<(usize, usize)> = order.iter().take(8).map(|&(x, y)| (y, x)).collect();
    let expected = [(3, 4), (4, 3), (0, 1), (1, 0), (1, 2), (2, 1), (2, 4), (3, 3)];
    assert_eq!(rc, expected, "B1 (b): the first eight pops, as (row, column)");
    assert_eq!(order.len(), 23, "B1 (d): 23 pops in total");
}

#[test]
fn c2_line_b1() {
    let (f, m) = b1();
    let (labels, order) = watershed(&f, &m, false, true, &mut HeapQueue::new());
    assert_eq!(line_pixels(&labels), [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)], "B1 (d): the line is the anti-diagonal (row, column)");
    assert_eq!((labels[(0, 0)], labels[(3, 0)], labels[(4, 1)], labels[(1, 4)]), (1, 1, 2, 2), "B1 (d): A above the line, B below");
    let first_line = order.iter().position(|&p| labels[p] == 0).unwrap();
    assert_eq!(first_line, 14, "B1 (d): the first line pixel is pop 15");
}

#[test]
fn c2_line_eight_b2() {
    let (f, m) = b1();
    let (labels, _) = watershed(&f, &m, true, true, &mut HeapQueue::new());
    assert_eq!(line_pixels(&labels), [(1, 3), (1, 4), (2, 2), (2, 3), (3, 1), (3, 2), (4, 1)], "B2: 7 line pixels with 8 neighbours");
}

#[test]
fn c2_every_pixel_labelled() {
    let (f, m) = b1();
    let (labels, _) = watershed(&f, &m, false, false, &mut HeapQueue::new());
    assert!(labels.pixels().iter().all(|&l| l == 1 || l == 2), "without a line, every pixel gets A or B:\n{labels}");
}

#[test]
fn c2_plateau_fifo_lifo_b3() {
    let (f, m) = signal(&[0., 5., 5., 5., 5., 5., 0.], &[(0, 1), (6, 2)]);
    let (fifo, _) = watershed(&f, &m, false, true, &mut HeapQueue::new());
    assert_eq!(fifo.pixels(), [1, 1, 1, 0, 2, 2, 2], "B3 (a): FIFO puts the line in the middle");
    let (lifo, _) = watershed(&f, &m, false, true, &mut HeapQueue::lifo());
    assert_eq!(lifo.pixels(), [1, 0, 2, 2, 2, 2, 2], "B3 (b): LIFO puts the line next to A");
}

#[test]
fn c2_marker_not_a_minimum_b4() {
    let (f, m) = signal(&[5., 3., 1., 2., 4., 6., 4., 2., 0., 2.], &[(0, 1), (8, 2)]);
    let (labels, order) = watershed(&f, &m, false, true, &mut HeapQueue::new());
    assert_eq!(labels.pixels(), [1, 1, 1, 1, 1, 0, 2, 2, 2, 2], "B4 (c): A A A A A | B B B B");
    let samples: Vec<usize> = order.iter().map(|p| p.0).collect();
    assert_eq!(samples, [7, 9, 1, 2, 3, 6, 4, 5], "B4 (b): the order of the pops (samples)");
}

#[test]
fn c3_bucket_queue_same_order_as_heap() {
    let mut rng = Rng::new(4);
    let (mut heap, mut buckets) = (HeapQueue::new(), BucketQueue::new());
    let mut popped = 0;
    for i in 0..2000 {
        // push mostly, pop sometimes, sometimes below the current level
        if rng.below(3) == 0 {
            let (a, b) = (heap.pop(), buckets.pop());
            assert_eq!(a, b, "pop {popped}: the heap gives {a:?}, the bucket queue {b:?}");
            popped += 1;
        } else {
            let h = rng.below(256) as f32;
            heap.push(h, (i, 0));
            buckets.push(h, (i, 0));
        }
    }
    while let Some(a) = heap.pop() {
        assert_eq!(Some(a), buckets.pop(), "emptying the queues: different pixels");
    }
    assert_eq!(buckets.pop(), None, "an empty bucket queue must return None");
}

#[test]
fn c3_bucket_watershed_same_as_heap() {
    let mut rng = Rng::new(9);
    let f = Image::from_fn(40, 30, |_, _| rng.below(256) as f32);
    let (m, _) = regional_minima(&f, true);
    let (a, _) = watershed(&f, &m, true, true, &mut HeapQueue::new());
    let (b, _) = watershed(&f, &m, true, true, &mut BucketQueue::new());
    assert!(a == b, "the bucket queue must give the same labels as the heap");
}
