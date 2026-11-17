// Checks for the coding exercises, using the label image of the pen-and-paper exercises.
// Run: cargo test -p labelling label_measure::

use imgcourse::Image;
use labelling::label_measure::*;

/// A label image written row by row ("." = 0).
fn labels(rows: &[&str]) -> Image<u32> {
    let cells: Vec<Vec<u32>> = rows.iter().map(|r| r.split_whitespace().map(|c| c.parse().unwrap_or(0)).collect()).collect();
    Image::from_fn(cells[0].len(), cells.len(), |x, y| cells[y][x])
}

fn b1() -> Image<u32> {
    labels(&[
        "1 1 1 . . . . 2 . .",
        "1 1 1 . 3 . . . . .",
        "1 1 1 . 3 . . . 4 .",
        ". . . . 3 3 . 4 4 4",
        "5 5 . . 3 3 . . 4 .",
        "5 5 . . . . . . . .",
        "5 5 . 6 6 6 6 . . 7",
        ". . . 6 6 6 6 . 7 7",
        "8 . . . . . . . 7 7",
        "8 8 . . . . . . . 7",
    ])
}

fn same(got: &Image<u32>, expected: &Image<u32>, what: &str) {
    for (x, y) in got.coords() {
        assert_eq!(got[(x, y)], expected[(x, y)], "{what}, row {y}, column {x}\ngot:\n{got}expected:\n{expected}");
    }
}

#[test]
fn c1_table_b1() {
    let t = table(&b1());
    let areas: Vec<usize> = t.iter().map(|r| r.area).collect();
    assert_eq!(areas, [9, 1, 6, 5, 6, 8, 6, 3], "B1: areas of labels 1..8");
    let border: Vec<bool> = t.iter().map(|r| r.touches_border(10, 10)).collect();
    assert_eq!(border, [true, true, false, true, true, false, true, true], "B1: touches border, labels 1..8");
    assert_eq!((t[2].x_min, t[2].x_max, t[2].y_min, t[2].y_max), (4, 5, 1, 4), "B1: bounding box of label 3 (x_min, x_max, y_min, y_max)");
}

#[test]
fn c2_relabel_b1() {
    // B1 (b): "area ≥ 5 and solidity > 0.9" keeps 1, 5 and 6.
    let mut keep = vec![false; 9];
    for k in [1, 5, 6] {
        keep[k] = true;
    }
    let (out, n) = keep_and_relabel(&b1(), &keep);
    assert_eq!(n, 3, "B1 (b): 3 objects kept");
    let expected = labels(&[
        "1 1 1 . . . . . . .",
        "1 1 1 . . . . . . .",
        "1 1 1 . . . . . . .",
        ". . . . . . . . . .",
        "2 2 . . . . . . . .",
        "2 2 . . . . . . . .",
        "2 2 . 3 3 3 3 . . .",
        ". . . 3 3 3 3 . . .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
    ]);
    same(&out, &expected, "B1 (b), relabelled");
}

#[test]
fn c2_relabel_follows_raster_order() {
    // Labels numbered against raster order: the result must be 1, 2, 3 from the top.
    let l = labels(&["3 . 1", ". . .", "2 2 ."]);
    let (out, n) = keep_and_relabel(&l, &[false, true, true, true]);
    same(&out, &labels(&["1 . 2", ". . .", "3 3 ."]), "relabel in raster order of the first pixel");
    assert_eq!(n, 3);
}

#[test]
fn c3_area_opening_b1() {
    let (out, n) = area_opening(&b1(), 5);
    assert_eq!(n, 6, "B1: area ≥ 5 keeps 1, 3, 4, 5, 6, 7: got {n} objects");
    assert_eq!(out[(7, 0)], 0, "label 2 (1 pixel) must be removed");
    assert_eq!(out[(0, 9)], 0, "label 8 (3 pixels) must be removed");
    assert_eq!(out[(4, 1)], 2, "label 3 becomes 2 (second in raster order)");
    // a connected filter keeps the survivors whole
    let kept = out.pixels().iter().filter(|&&l| l > 0).count();
    assert_eq!(kept, 9 + 6 + 5 + 6 + 8 + 6, "the kept objects must keep all their pixels");
}

#[test]
fn c4_counting_frame_b2() {
    let counted = counting_frame(&b1(), 2);
    assert_eq!(counted, vec![1, 3, 6], "B2: counted labels with margin 2");
}

#[test]
fn c4_frames_count_each_object_once() {
    // Four frames of 40 × 40 tile the square [20, 100)²; each is counted in its own 80 × 80
    // window (guard zone 20, larger than any object). Every object inside the tiled square must
    // be counted by exactly one of the four frames, whatever its size.
    let mut rng = imgcourse::Rng::new(3);
    let (w, h) = (120, 120);
    let mut img = Image::new(w, h, 0u32);
    for k in 1..=60u32 {
        let (cx, cy) = (rng.below(w), rng.below(h));
        let r = 1 + rng.below(6);
        for (x, y) in img.coords() {
            if x.abs_diff(cx).max(y.abs_diff(cy)) <= r && img[(x, y)] == 0 {
                img[(x, y)] = k;
            }
        }
    }
    let mut times = vec![0; 61];
    for (ox, oy) in [(0, 0), (40, 0), (0, 40), (40, 40)] {
        for l in counting_frame(&img.crop(ox, oy, 80, 80), 20) {
            times[l as usize] += 1;
        }
    }
    let rows = table(&img);
    for (k, r) in rows.iter().enumerate() {
        let inside = r.area > 0 && r.x_min >= 21 && r.x_max <= 98 && r.y_min >= 21 && r.y_max <= 98;
        if inside {
            assert_eq!(times[k + 1], 1, "object {} (box x {}..{}, y {}..{}) was counted {} times", k + 1, r.x_min, r.x_max, r.y_min, r.y_max, times[k + 1]);
        }
        assert!(times[k + 1] <= 1, "object {} was counted by two frames", k + 1);
    }
}

#[test]
fn c5_hysteresis() {
    // Three blobs on one row: bright (0.9), dim (0.4) and dim with one bright pixel.
    let img = Image::from_rows(&[[0.0, 0.9, 0.9, 0.0, 0.4, 0.4, 0.0, 0.4, 0.8, 0.4, 0.0]]);
    let (out, n) = hysteresis(&img, 0.3, 0.7);
    assert_eq!(n, 2, "low 0.3, high 0.7: two of the three blobs contain a bright pixel; got {n}");
    let got: Vec<u32> = out.pixels().to_vec();
    assert_eq!(got, [0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0], "the kept blobs are kept whole, the dim one goes");
}
