// Skeleton graphs: the length estimators on digital straight lines, and a thick T with a
// bumpy boundary thinned, turned into a graph and pruned. Numbers printed, pictures in
// out/skel-graph/.
//   cargo run -p skel-graph                      (your code)
//   cargo run -p skel-graph --features solution  (the model solution)

use imgcourse::*;
use skel_graph::*;

/// The 8 neighbours x0..x7 of a pixel, going round from the top: N, NE, E, SE, S, SW, W, NW.
fn ring(img: &Image<bool>, x: usize, y: usize) -> [bool; 8] {
    let offsets = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)];
    offsets.map(|(dx, dy)| img.get_border(x as isize + dx, y as isize + dy, Border::Zero))
}

/// Thinning (Zhang & Suen 1984, the lecture "Hit-or-miss and thinning"), followed by the
/// removal of staircase corners (a pixel with two object 4-neighbours at a right angle that
/// is simple), so that the skeleton is one pixel thin in the 8-neighbourhood.
fn thin(mask: &Image<bool>) -> Image<bool> {
    let mut img = mask.clone();
    loop {
        let mut changed = false;
        for sub in 0..2 {
            let remove = img.map_xy(|x, y, v| {
                if !v {
                    return false;
                }
                let n = ring(&img, x, y);
                let count = n.iter().filter(|&&b| b).count();
                let transitions = (0..8).filter(|&i| !n[i] && n[(i + 1) % 8]).count();
                let (p2, p4, p6, p8) = (n[0], n[2], n[4], n[6]);
                let side = if sub == 0 { !(p2 && p4 && p6) && !(p4 && p6 && p8) } else { !(p2 && p4 && p8) && !(p2 && p6 && p8) };
                (2..=6).contains(&count) && transitions == 1 && side
            });
            if remove.pixels().iter().any(|&r| r) {
                changed = true;
                img = img.zip_map(&remove, |v, r| v && !r);
            }
        }
        if !changed {
            break;
        }
    }
    remove_corners(&mut img);
    img
}

/// Remove staircase corners, one at a time, where that keeps the topology (Yokoi's
/// connectivity number is 1): a corner pixel has two object 4-neighbours at a right angle,
/// which are already connected diagonally. Without this, the three pixels of a staircase
/// step would all count as junction pixels.
fn remove_corners(img: &mut Image<bool>) {
    for (x, y) in img.coords().collect::<Vec<_>>() {
        if !img[(x, y)] {
            continue;
        }
        let n = ring(img, x, y);
        let corner = (0..4).any(|k| n[2 * k] && n[(2 * k + 2) % 8]);
        let b = n.map(|v| !v as i32);
        let yokoi: i32 = [0, 2, 4, 6].iter().map(|&k| b[k] - b[k] * b[(k + 1) % 8] * b[(k + 2) % 8]).sum();
        let count = n.iter().filter(|&&v| v).count();
        if corner && yokoi == 1 && count >= 2 {
            img[(x, y)] = false;
        }
    }
}

/// A thick T (half-width 16) with small round bumps on its boundary, 256 × 256.
fn bumpy_t() -> Image<bool> {
    let mut rng = Rng::new(26);
    let inside_t = |x: f32, y: f32| ((40.0..216.0).contains(&x) && (48.0..80.0).contains(&y)) || ((112.0..144.0).contains(&x) && (48.0..220.0).contains(&y));
    // bumps: discs of radius 3 to 7 centred on boundary points of the T
    let mut bumps = Vec::new();
    while bumps.len() < 26 {
        let (x, y) = (256.0 * rng.uniform(), 256.0 * rng.uniform());
        let on_edge = inside_t(x, y) && [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)].iter().any(|(dx, dy)| !inside_t(x + dx, y + dy));
        if on_edge {
            bumps.push((x, y, 3.0 + 4.0 * rng.uniform()));
        }
    }
    Image::from_fn(256, 256, |x, y| {
        let (x, y) = (x as f32, y as f32);
        inside_t(x, y) || bumps.iter().any(|&(bx, by, r)| (x - bx).powi(2) + (y - by).powi(2) <= r * r)
    })
}

fn describe(name: &str, skel: &Image<bool>) -> Graph {
    let g = build_graph(skel);
    let mut degree = vec![0; g.nodes.len()];
    for e in &g.edges {
        degree[e.a] += 1;
        degree[e.b] += 1;
    }
    let ends = degree.iter().filter(|&&d| d == 1).count();
    let junctions = degree.iter().filter(|&&d| d >= 3).count();
    let total = |f: fn(&Steps) -> f32| g.edges.iter().map(|e| f(&path_steps(&e.path))).sum::<f32>();
    println!(
        "{name:<18} V {:>3}  E {:>3}  ends {:>3}  junctions {:>3}  E−V+C {:>2}  length: steps {:>6.1}, 1/√2 {:>6.1}, corrected {:>6.1}",
        g.nodes.len(),
        g.edges.len(),
        ends,
        junctions,
        cycles(&g),
        total(Steps::pixel_steps),
        total(Steps::weighted),
        total(Steps::corrected)
    );
    g
}

/// Mask grey, removed skeleton pixels pink, edges yellow and green in turn, end points blue,
/// junctions red.
fn picture(mask: &Image<bool>, before: &Image<bool>, g: &Graph) -> Image<[f32; 3]> {
    let mut pic = mask.map_xy(|x, y, m| if before[(x, y)] { [1.0, 0.4, 0.8] } else if m { [0.3; 3] } else { [0.0; 3] });
    for (i, e) in g.edges.iter().enumerate() {
        let colour = if i % 2 == 0 { [1.0, 0.85, 0.1] } else { [0.2, 0.9, 0.3] };
        for &p in &e.path {
            pic[p] = colour;
        }
    }
    let mut degree = vec![0; g.nodes.len()];
    for e in &g.edges {
        degree[e.a] += 1;
        degree[e.b] += 1;
    }
    for (n, d) in g.nodes.iter().zip(degree) {
        for &p in &n.pixels {
            pic[p] = if d == 1 { [0.2, 0.5, 1.0] } else { [1.0, 0.1, 0.1] };
        }
    }
    // enlarge twice
    Image::from_fn(pic.width() * 2, pic.height() * 2, |x, y| pic[(x / 2, y / 2)])
}

fn main() {
    let dir = "out/skel-graph";

    // 1. Digital straight lines of about 170 pixels at different angles.
    println!("length of a digital straight line, estimate / true:");
    println!("angle   true   steps   1/√2   corrected   (ne, no, nc)");
    for deg in [0, 10, 20, 22, 26, 30, 40, 45] {
        let t = (deg as f32).to_radians().tan();
        let n = (170.0 * (deg as f32).to_radians().cos()).round() as usize;
        let path: Vec<(usize, usize)> = (0..=n).map(|x| (x, (x as f32 * t).round() as usize)).collect();
        let (a, b) = (path[0], path[n]);
        let truth = ((b.0 as f32 - a.0 as f32).powi(2) + (b.1 as f32 - a.1 as f32).powi(2)).sqrt();
        let s = path_steps(&path);
        println!(
            "{deg:>4}°  {truth:>6.1}  {:>5.3}  {:>5.3}  {:>5.3}       ({}, {}, {})",
            s.pixel_steps() / truth,
            s.weighted() / truth,
            s.corrected() / truth,
            s.ne,
            s.no,
            s.nc
        );
    }

    // 2. The bumpy T: skeleton, graph, pruning.
    println!();
    let mask = bumpy_t();
    let skel = thin(&mask);
    let g = describe("T, not pruned", &skel);
    save_rgb(&picture(&mask, &Image::new(256, 256, false), &g), format!("{dir}/t-graph.png"));
    for k in [12.0, 24.0] {
        let mut pruned = prune(&skel, k);
        remove_corners(&mut pruned); // a pruned junction can leave a staircase step behind
        let g = describe(&format!("T, pruned k = {k}"), &pruned);
        let removed = skel.zip_map(&pruned, |a, b| a && !b);
        save_rgb(&picture(&mask, &removed, &g), format!("{dir}/t-pruned-{k}.png"));
    }

    // 3. A plate with three holes: cycles = holes.
    let plate = Image::from_fn(256, 256, |x, y| {
        let hole = |cx: f32, cy: f32, r: f32| (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) < r * r;
        (40..216).contains(&x) && (40..216).contains(&y) && !hole(90.0, 90.0, 25.0) && !hole(165.0, 100.0, 20.0) && !hole(120.0, 165.0, 28.0)
    });
    let plate_skel = thin(&plate);
    let g = describe("plate, 3 holes", &plate_skel);
    save_rgb(&picture(&plate, &Image::new(256, 256, false), &g), format!("{dir}/plate-graph.png"));
    println!("wrote the pictures to {dir}/ (edges yellow and green in turn, blue: end points, red: junctions, pink: pruned)");
}
