// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p hough-lines

use hough_lines::sim::EdgePoint;
use hough_lines::*;
use std::f32::consts::PI;

#[test]
fn c1_accumulator_b1() {
    // B1: rows θ = 0°, 45°, 90°, 135°, columns ρ = −2 … 2. Here θ is the column j, ρ the row k.
    let acc = accumulate(&[[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]], 4, 1.0, 2.0);
    assert_eq!((acc.width(), acc.height()), (4, 5), "4 angles, 5 distances");
    let expected = [
        [0., 0., 1., 1., 1.], // 0°
        [0., 0., 0., 3., 0.], // 45°
        [0., 0., 1., 1., 1.], // 90°
        [0., 1., 1., 1., 0.], // 135°
    ];
    for j in 0..4 {
        let row: Vec<f32> = (0..5).map(|k| acc[(j, k)]).collect();
        assert_eq!(row, expected[j], "B1(b): votes at θ = {}° for ρ = −2 … 2", 45 * j);
    }
}

#[test]
fn c1_dashed_line_b2a() {
    // a horizontal line y = 5 of 120 points, and the same line dashed (10 on, 10 off)
    let full: Vec<[f32; 2]> = (0..120).map(|x| [x as f32 - 60.0, 5.0]).collect();
    let dashed: Vec<[f32; 2]> = full.iter().copied().filter(|p| ((p[0] + 60.0) as i32 / 10) % 2 == 0).collect();
    let a = accumulate(&full, 180, 1.0, 90.0);
    let b = accumulate(&dashed, 180, 1.0, 90.0);
    // the line y = 5 is θ = 90°, ρ = 5: column 90, row 95
    assert_eq!(a[(90, 95)], 120.0, "full line: expected 120 votes at θ = 90°, ρ = 5");
    assert_eq!(b[(90, 95)], 60.0, "B2(a): dashed line: expected 60 votes in the same bin");
    assert_eq!(a.sum(), 120.0 * 180.0, "every point votes once per angle: 120 × 180 votes");
}

/// Two parallel vertical lines x = 10 and x = 13 (θ = 0), 100 points each.
fn b3_points() -> Vec<[f32; 2]> {
    let mut p = Vec::new();
    for y in 0..100 {
        p.push([10.0, y as f32 - 50.0]);
        p.push([13.0, y as f32 - 50.0]);
    }
    p
}

#[test]
fn c3_parallel_lines_b3() {
    // Δρ = 4 with bin centres 10, 14 (bins [8, 12), [12, 16)): two neighbouring bins, one peak
    let acc = accumulate(&b3_points(), 36, 4.0, 30.0);
    assert!(acc[(0, 10)] == 100.0 && acc[(0, 11)] == 100.0, "B3(a): 100 votes each at ρ = 10 and 14");
    let p = peaks(&acc, 50.0, 1);
    assert_eq!(p.len(), 1, "B3(b): with NMS ±1 bin only one of the two equal bins is kept, got {p:?}");
    // bin centre 12 (bin [10, 14)): both lines in one bin
    let acc = accumulate(&b3_points(), 36, 4.0, 28.0);
    assert_eq!(acc[(0, 10)], 200.0, "B3(a): both lines in the bin [10, 14)");
    // Δρ = 1: two peaks
    let acc = accumulate(&b3_points(), 36, 1.0, 60.0);
    let p = peaks(&acc, 50.0, 1);
    assert_eq!(p.len(), 2, "B3(b): with Δρ = 1 and NMS ±1 both lines are found, got {p:?}");
}

#[test]
fn c3_wraps_around() {
    // A nearly vertical line, normal at 179°: its votes near θ = 0 have ρ of opposite sign.
    let a = 179f32.to_radians();
    let (n, d) = ([a.cos(), a.sin()], [-a.sin(), a.cos()]);
    let pts: Vec<[f32; 2]> = (0..80).map(|t| {
        let t = t as f32 - 40.0;
        [-20.0 * n[0] + t * d[0], -20.0 * n[1] + t * d[1]]
    }).collect();
    let acc = accumulate(&pts, 180, 1.0, 60.0);
    let p = peaks(&acc, 30.0, 3);
    assert_eq!(p.len(), 1, "one line across the 0°/180° seam should give one peak, got {p:?}");
    assert_eq!((p[0].0, p[0].1), (179, 40), "peak at θ = 179°, ρ = −20 (row 40)");
}

#[test]
fn c2_gradient_voting_b4() {
    // points on the line y = 5 with gradient pointing down (θ = 90°), window ±2 bins
    let pts: Vec<[f32; 2]> = (0..50).map(|x| [x as f32 - 25.0, 5.0]).collect();
    let edges: Vec<EdgePoint> = pts.iter().map(|p| EdgePoint { x: p[0], y: p[1], theta: PI / 2.0, strength: 1.0 }).collect();
    let acc = accumulate_gradient(&pts, &edges, 180, 1.0, 60.0, 2);
    assert_eq!(acc.sum(), 50.0 * 5.0, "B4(c): each point casts 2 · 2 + 1 = 5 votes");
    assert_eq!(acc[(90, 65)], 50.0, "the peak θ = 90°, ρ = 5 keeps all 50 votes");
    // the gradient may point the other way (θ = 270°): the same columns, modulo 180°
    let flipped: Vec<EdgePoint> = edges.iter().map(|e| EdgePoint { theta: -PI / 2.0, ..*e }).collect();
    let acc2 = accumulate_gradient(&pts, &flipped, 180, 1.0, 60.0, 2);
    assert_eq!(acc2[(90, 65)], 50.0, "gradient −90° is the same line direction as 90°");
    // near 0°: columns 178, 179, 0, 1, 2
    let e0 = [EdgePoint { x: 0.0, y: 0.0, theta: 0.0, strength: 1.0 }];
    let acc3 = accumulate_gradient(&[[3.0, 0.0]], &e0, 180, 1.0, 60.0, 2);
    let cols: Vec<usize> = (0..180).filter(|&j| (0..acc3.height()).any(|k| acc3[(j, k)] > 0.0)).collect();
    assert_eq!(cols, vec![0, 1, 2, 178, 179], "the window wraps around at 0°/180°");
}

#[test]
fn c4_segments_gaps() {
    // a dashed horizontal line: dashes of 10 px (11 points), gaps of 10 px, from x = 0 to 70
    let mut pts = Vec::new();
    for start in [0, 20, 40, 60] {
        for x in start..=start + 10 {
            pts.push([x as f32, 3.0]);
        }
    }
    pts.push([30.0, 40.0]); // far from the line
    let theta = PI / 2.0; // the line y = 3
    assert!(segments(&pts, theta, 3.0, 1.0, 6.0, 12.0).is_empty(), "gap 6, minimum 12: every dash is too short");
    let s = segments(&pts, theta, 3.0, 1.0, 6.0, 8.0);
    assert_eq!(s.len(), 4, "gap 6, minimum 8: four dashes, got {s:?}");
    let s = segments(&pts, theta, 3.0, 1.0, 12.0, 12.0);
    assert_eq!(s.len(), 1, "gap 12: the dashes join into one segment, got {s:?}");
    let (a, b) = s[0];
    assert!((a[0] - b[0]).abs() == 70.0, "the joined segment runs from x = 0 to 70, got {a:?} to {b:?}");
}
