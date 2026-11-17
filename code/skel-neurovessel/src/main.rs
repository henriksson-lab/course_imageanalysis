// A synthetic traced neuron (a tree file: points with radius and parent): neuron
// measurements, Strahler orders, the Sholl profile, tortuosity, and matching branch points
// against a disturbed "measurement". Numbers printed; a picture in out/skel-neurovessel/.
//   cargo run -p skel-neurovessel                      (your code)
//   cargo run -p skel-neurovessel --features solution  (the model solution)

use imgcourse::*;
use skel_neurovessel::*;

const SIZE: usize = 256;
const SOMA: (f32, f32) = (128.0, 128.0);

/// Grow a dendrite from point `from` in direction `angle`: steps of 3 pixels with a little
/// random turning; after 14 to 32 pixels it either ends or splits in two (radius × 0.794,
/// Murray's law for a symmetric split), at most `depth` more times.
fn grow(tree: &mut Vec<Point>, rng: &mut Rng, from: usize, mut angle: f32, radius: f32, depth: usize) {
    let length = 14.0 + 18.0 * rng.uniform();
    let mut current = from;
    let mut done = 0.0;
    while done < length {
        angle += 0.12 * rng.normal();
        let (x, y) = (tree[current].x + 3.0 * angle.cos(), tree[current].y + 3.0 * angle.sin());
        if !(4.0..SIZE as f32 - 4.0).contains(&x) || !(4.0..SIZE as f32 - 4.0).contains(&y) {
            return; // stop at the image border
        }
        tree.push(Point { x, y, radius, parent: Some(current) });
        current = tree.len() - 1;
        done += 3.0;
    }
    if depth > 0 && rng.uniform() < 0.85 {
        let spread = 0.35 + 0.3 * rng.uniform();
        for side in [-1.0, 1.0] {
            grow(tree, rng, current, angle + side * spread, radius * 0.794, depth - 1);
        }
    }
}

fn neuron(seed: u64) -> Vec<Point> {
    let mut rng = Rng::new(seed);
    let mut tree = vec![Point { x: SOMA.0, y: SOMA.1, radius: 8.0, parent: None }];
    for k in 0..5 {
        let angle = 2.0 * std::f32::consts::PI * (k as f32 + 0.3 * rng.uniform()) / 5.0;
        // start on the edge of the soma
        tree.push(Point { x: SOMA.0 + 8.0 * angle.cos(), y: SOMA.1 + 8.0 * angle.sin(), radius: 3.0, parent: Some(0) });
        let first = tree.len() - 1;
        grow(&mut tree, &mut rng, first, angle, 3.0, 3);
    }
    tree
}

/// Draw the tree as tubes coloured by Strahler order, the branch points in white and the
/// Sholl circle of radius `r` in grey.
fn picture(tree: &[Point], order: &[u32], r: f32) -> Image<[f32; 3]> {
    let colours = [[0.3, 0.6, 1.0], [0.2, 0.9, 0.4], [1.0, 0.8, 0.1], [1.0, 0.3, 0.2], [0.9, 0.3, 1.0]];
    let kids = children(tree);
    Image::from_fn(SIZE, SIZE, |x, y| {
        let (px, py) = (x as f32, y as f32);
        let near = |p: &Point, d: f32| (px - p.x).powi(2) + (py - p.y).powi(2) <= d * d;
        if tree[0].parent.is_none() && near(&tree[0], tree[0].radius) {
            return [0.8, 0.8, 0.8];
        }
        for (i, p) in tree.iter().enumerate() {
            if p.parent.is_some() && kids[i].len() >= 2 && near(p, 2.5) {
                return [1.0, 1.0, 1.0];
            }
        }
        for (i, p) in tree.iter().enumerate() {
            if let Some(q) = p.parent {
                // distance from the pixel to the segment q–p
                let (ax, ay, dx, dy) = (tree[q].x, tree[q].y, p.x - tree[q].x, p.y - tree[q].y);
                let t = (((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                if (px - ax - t * dx).powi(2) + (py - ay - t * dy).powi(2) <= p.radius.max(0.8).powi(2) {
                    return colours[(order[i] as usize - 1).min(4)];
                }
            }
        }
        let dc = ((px - SOMA.0).powi(2) + (py - SOMA.1).powi(2)).sqrt();
        if (dc - r).abs() < 0.6 { [0.45, 0.45, 0.45] } else { [0.0; 3] }
    })
}

fn main() {
    let dir = "out/skel-neurovessel";
    let tree = neuron(2);
    let (length, branch_points, tips) = tree_counts(&tree);
    let primary = children(&tree)[0].len();
    println!("synthetic neuron: {} points, total length {length:.0} px, {primary} primary branches, {branch_points} branch points, {tips} tips", tree.len());

    let order = strahler(&tree);
    let br = branches(&tree);
    let max_order = br.iter().map(|b| order[*b.points.last().unwrap()]).max().unwrap_or(0);
    print!("branches per Strahler order:");
    for s in 1..=max_order {
        print!("  order {s}: {}", br.iter().filter(|b| order[*b.points.last().unwrap()] == s).count());
    }
    let mean_tort = br.iter().map(|b| b.tortuosity()).sum::<f32>() / br.len() as f32;
    let max_tort = br.iter().map(|b| b.tortuosity()).fold(0.0, f32::max);
    println!("\n{} branches, mean tortuosity {mean_tort:.3}, largest {max_tort:.3}", br.len());

    let radii: Vec<f32> = (1..=13).map(|k| 10.0 * k as f32).collect();
    let profile = sholl(&tree, SOMA.0, SOMA.1, &radii);
    print!("Sholl profile N(r):");
    for (r, n) in radii.iter().zip(&profile) {
        print!("  {r}: {n}");
    }
    let (best, &most) = profile.iter().enumerate().max_by_key(|&(i, n)| (n, std::cmp::Reverse(i))).unwrap();
    println!("\nmaximum {most} crossings at r = {}", radii[best]);

    // A "measurement" of the branch points: the true ones moved by about 3 pixels, one lost,
    // two spurious ones added. Count matches at several tolerances.
    let kids = children(&tree);
    let truth: Vec<[f32; 2]> = (1..tree.len()).filter(|&i| kids[i].len() >= 2).map(|i| [tree[i].x, tree[i].y]).collect();
    let mut rng = Rng::new(3);
    let mut measured: Vec<[f32; 2]> = truth.iter().skip(1).map(|p| [p[0] + 3.0 * rng.normal(), p[1] + 3.0 * rng.normal()]).collect();
    measured.push([40.0, 200.0]);
    measured.push([200.0, 50.0]);
    println!("matching {} true and {} measured branch points:", truth.len(), measured.len());
    for tol in [2.0, 4.0, 8.0] {
        let (m, missed, spurious) = match_points(&truth, &measured, tol);
        println!(
            "  tolerance {tol}: {} matched, {} missed, {} spurious; precision {:.2}, recall {:.2}",
            m.len(),
            missed.len(),
            spurious.len(),
            m.len() as f32 / measured.len() as f32,
            m.len() as f32 / truth.len() as f32
        );
    }
    save_rgb(&picture(&tree, &order, radii[best]), format!("{dir}/neuron.png"));
    println!("wrote {dir}/neuron.png (colour = Strahler order: blue 1, green 2, yellow 3, red 4; grey circle at the Sholl maximum)");
}
