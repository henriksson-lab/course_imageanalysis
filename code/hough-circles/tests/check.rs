// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p hough-circles

use hough_circles::sim::{edge_points, ellipse, polygon_scene, EdgePoint};
use hough_circles::*;

fn point(x: f32, y: f32, ux: f32, uy: f32) -> EdgePoint {
    EdgePoint { x, y, theta: uy.atan2(ux), strength: 1.0 }
}

/// The pixels of a digital ring of radius r around (cx, cy): |distance − r| < 1/2.
fn ring(cx: isize, cy: isize, r: f32) -> Vec<EdgePoint> {
    let mut v = Vec::new();
    let reach = r as isize + 1;
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if (d - r).abs() < 0.5 {
                v.push(point((cx + dx) as f32, (cy + dy) as f32, -dx as f32, -dy as f32));
            }
        }
    }
    v
}

#[test]
fn c2_gradient_votes_b1() {
    // B1(a): radius 5, point (3, 4), gradient (0.6, 0.8): centres (6, 8) and (0, 0)
    let a = gradient_votes(&[point(3.0, 4.0, 0.6, 0.8)], 12, 12, 5.0, Polarity::Either);
    assert!(a[(6, 8)] == 1.0 && a[(0, 0)] == 1.0 && a.sum() == 2.0, "B1(a): votes at (6, 8) and (0, 0) expected");
    // B1(b): three points of a bright disc, both polarities
    let pts = [point(15.0, 10.0, -1.0, 0.0), point(10.0, 15.0, 0.0, -1.0), point(13.0, 14.0, -0.6, -0.8)];
    let b = gradient_votes(&pts, 25, 25, 5.0, Polarity::Either);
    assert_eq!(b[(10, 10)], 3.0, "B1(b): the centre (10, 10) should get 3 votes");
    assert_eq!(b.sum(), 6.0, "B1(b): 6 votes in total");
    let c = gradient_votes(&pts, 25, 25, 5.0, Polarity::BrightOnDark);
    assert!(c[(10, 10)] == 3.0 && c.sum() == 3.0, "B1(c): bright disc: 3 votes, all at (10, 10)");
    let d = gradient_votes(&pts, 25, 25, 5.0, Polarity::DarkOnBright);
    assert_eq!(d[(10, 10)], 0.0, "dark on bright: no vote reaches the centre of a bright disc");
}

#[test]
fn c1_ring_votes_b3() {
    let pts = ring(20, 20, 10.0);
    let acc = ring_votes(&pts, 41, 41, 10.0);
    assert_eq!(acc[(20, 20)], pts.len() as f32, "every point of the ring votes for its centre");
    assert!((50..=64).contains(&pts.len()), "a digital ring of radius 10 has about 2π·10 = 63 pixels, a few fewer; got {}", pts.len());
    assert_eq!(acc.max(), acc[(20, 20)], "the centre is the largest bin");
    // B3: a quarter hidden (the points with x > 20 and y > 20 removed)
    let three_quarters: Vec<EdgePoint> = pts.iter().copied().filter(|p| !(p.x > 20.0 && p.y > 20.0)).collect();
    let acc = ring_votes(&three_quarters, 41, 41, 10.0);
    let frac = acc[(20, 20)] / pts.len() as f32;
    assert!((0.7..0.8).contains(&frac), "B3: a quarter hidden: expected about 0.75 of the votes, got {frac}");
}

#[test]
fn c1_touching_discs_b4() {
    let mut pts = ring(15, 20, 10.0);
    pts.extend(ring(35, 20, 10.0));
    let acc = ring_votes(&pts, 50, 41, 10.0);
    let contact = acc[(25, 20)];
    let centre = acc[(15, 20)];
    assert!(contact <= 8.0 && centre >= 50.0, "B4(a): the contact point gets a few votes ({contact}), each centre about 56 ({centre})");
}

#[test]
fn c3_find_circles() {
    // a bright disc of radius 9 at (30, 25) and a dark one of radius 6 at (12, 12) on grey
    let img = polygon_scene(48, 0.4, &[(ellipse(30.0, 25.0, 9.0, 9.0, 0.0), 0.9), (ellipse(12.0, 12.0, 6.0, 6.0, 0.0), 0.0)], 0.0, 1);
    let edges = edge_points(&img, 1.0, 0.2);
    let radii: Vec<f32> = (4..=12).map(|r| r as f32).collect();
    let slices: Vec<_> = radii.iter().map(|&r| ring_votes(&edges, 48, 48, r)).collect();
    let found = find_circles(&slices, &radii, 0.5);
    assert_eq!(found.len(), 2, "two discs, two circles above 0.5; got {found:?}");
    let near = |c: &Circle, x: f32, y: f32, r: f32| (c.x as f32 - x).abs() <= 1.0 && (c.y as f32 - y).abs() <= 1.0 && (c.r - r).abs() <= 1.0;
    assert!(found.iter().any(|c| near(c, 30.0, 25.0, 9.0)), "disc (30, 25) r = 9 not found: {found:?}");
    assert!(found.iter().any(|c| near(c, 12.0, 12.0, 6.0)), "disc (12, 12) r = 6 not found: {found:?}");
    // gradient voting with the bright polarity finds only the bright disc
    let slices: Vec<_> = radii.iter().map(|&r| gradient_votes(&edges, 48, 48, r, Polarity::BrightOnDark)).collect();
    let found = find_circles(&slices, &radii, 0.2);
    assert!(found.iter().any(|c| near(c, 30.0, 25.0, 9.0)) && !found.iter().any(|c| near(c, 12.0, 12.0, 6.0)), "bright on dark: only (30, 25); got {found:?}");
}
