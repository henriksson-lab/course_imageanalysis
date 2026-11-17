// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day39 detection::

use day39::detection::*;
use imgcourse::Image;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

/// The five boxes of B2.
fn five() -> Vec<Detection> {
    vec![
        Detection::new(0.0, 10.0, 0.0, 10.0, 0.9),
        Detection::new(1.0, 11.0, 0.0, 10.0, 0.8),
        Detection::new(8.0, 18.0, 0.0, 10.0, 0.7),
        Detection::new(9.0, 19.0, 1.0, 11.0, 0.6),
        Detection::new(20.0, 26.0, 20.0, 26.0, 0.3),
    ]
}

#[test]
fn c1_iou_b1() {
    let a = Detection::new(0.0, 4.0, 0.0, 4.0, 1.0);
    for (b, expected, what) in [
        (Detection::new(2.0, 6.0, 2.0, 6.0, 1.0), 4.0 / 28.0, "B1 (a): [2, 6]²"),
        (Detection::new(1.0, 5.0, 0.0, 4.0, 1.0), 0.6, "B1 (b): shifted right by 1"),
        (Detection::new(1.0, 5.0, 1.0, 5.0, 1.0), 9.0 / 23.0, "B1 (b): shifted right and down by 1"),
        (Detection::new(10.0, 14.0, 0.0, 4.0, 1.0), 0.0, "no overlap"),
        (a, 1.0, "the same box"),
    ] {
        let got = iou(&a, &b);
        assert!(close(got, expected), "{what}: got {got}, expected {expected}");
    }
    let small = iou(&Detection::new(0.0, 2.0, 0.0, 2.0, 1.0), &Detection::new(1.0, 3.0, 0.0, 2.0, 1.0));
    assert!(close(small, 1.0 / 3.0), "B1 (c): two 2 × 2 boxes shifted by one: got {small}, expected 0.33");
}

#[test]
fn c1_iou_pairs_b2a() {
    let b = five();
    for (i, j, expected) in [(0, 1, 0.818), (0, 2, 0.111), (0, 3, 0.047), (1, 2, 0.176), (2, 3, 0.681)] {
        let got = iou(&b[i], &b[j]);
        assert!(close(got, expected), "B2 (a): IoU of b{} and b{}: got {got}, expected {expected}", i + 1, j + 1);
    }
}

#[test]
fn c2_nms_b2() {
    let b = five();
    for (t, expected) in [(0.3, vec![0, 2, 4]), (0.7, vec![0, 2, 3, 4]), (0.1, vec![0, 3, 4])] {
        let kept = nms(&b, t);
        let names: Vec<String> = kept.iter().map(|i| format!("b{}", i + 1)).collect();
        let want: Vec<String> = expected.iter().map(|i| format!("b{}", i + 1)).collect();
        assert_eq!(names, want, "B2: NMS with t = {t}");
    }
    // The order of the input does not matter, only the scores.
    let mut shuffled = five();
    shuffled.reverse();
    let kept: Vec<f32> = nms(&shuffled, 0.3).iter().map(|&i| shuffled[i].score).collect();
    assert_eq!(kept, vec![0.9, 0.7, 0.3], "NMS on the boxes in reverse order: kept scores");
}

#[test]
fn c3_boxes_from_labels() {
    let mut labels = Image::new(10, 8, 0u32);
    labels[(2, 1)] = 1;
    labels[(4, 3)] = 1;
    labels[(7, 6)] = 2;
    let boxes = boxes_from_labels(&labels);
    assert_eq!(boxes.len(), 2, "two labels, two boxes");
    let (a, b) = (boxes[0], boxes[1]);
    assert!((a.x0, a.x1, a.y0, a.y1) == (2.0, 5.0, 1.0, 4.0), "label 1 at (2, 1) and (4, 3): got {a:?}, expected [2, 5] × [1, 4]");
    assert!((b.x0, b.x1, b.y0, b.y1) == (7.0, 8.0, 6.0, 7.0), "label 2, one pixel at (7, 6): got {b:?}, expected [7, 8] × [6, 7]");
}

#[test]
fn c4_count_matches_b2b() {
    let b = five();
    let truth = vec![b[0], b[2]]; // the true cells lie under b1 and b3
    for (t, expected) in [(0.3, (2, 1, 0)), (0.7, (2, 2, 0)), (0.1, (2, 1, 0))] {
        let kept: Vec<Detection> = nms(&b, t).iter().map(|&i| b[i]).collect();
        let got = count_matches(&kept, &truth, 0.5);
        assert_eq!(got, expected, "B2: (TP, FP, FN) after NMS with t = {t}");
    }
    let got = count_matches(&b, &truth, 0.5);
    assert_eq!(got, (2, 3, 0), "without NMS: b2 and b4 are duplicates (FP), b5 is false");
}
