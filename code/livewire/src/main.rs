// Traces a synthetic nucleus with your live-wire; results in out/livewire/.
//   cargo run -p livewire                      (your code)
//   cargo run -p livewire --features solution  (the model solution)

use imgcourse::*;
use livewire::*;

/// Gaussian smoothing in two passes (from the filtering unit; not part of these exercises).
fn smooth(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}

/// The length of a pixel path: steps of 1 and √2.
fn length(path: &[(usize, usize)]) -> f32 {
    path.windows(2).map(|p| if p[0].0 != p[1].0 && p[0].1 != p[1].1 { 2f32.sqrt() } else { 1.0 }).sum()
}

fn draw(rgb: &mut Image<[f32; 3]>, path: &[(usize, usize)], colour: [f32; 3]) {
    for &p in path {
        rgb[p] = colour;
    }
}

fn main() {
    let dir = "out/livewire";
    let (img, labels) = samples::nuclei(512, 14, 7);
    let cost = edge_cost(&smooth(&img, 1.5));
    save_gray(&img, format!("{dir}/input.png"));
    save_gray_stretched(&cost, format!("{dir}/cost.png"));

    // Four anchors on the true boundary of nucleus 14: the last pixel of the nucleus along
    // the rays to the right, down, left and up from its centre.
    let label = 14;
    let inside: Vec<(usize, usize)> = labels.coords().filter(|&p| labels[p] == label).collect();
    let cx = inside.iter().map(|p| p.0).sum::<usize>() / inside.len();
    let cy = inside.iter().map(|p| p.1).sum::<usize>() / inside.len();
    let edge = |dx: isize, dy: isize| {
        let (mut x, mut y) = (cx as isize, cy as isize);
        while labels[((x + dx) as usize, (y + dy) as usize)] == label {
            x += dx;
            y += dy;
        }
        (x as usize, y as usize)
    };
    let anchors = [edge(1, 0), edge(0, 1), edge(-1, 0), edge(0, -1)];
    println!("nucleus {label}: centre ({cx}, {cy}), anchors {anchors:?}");

    // One Dijkstra run per anchor; the wire to the next anchor by backtracking.
    let mut rgb = gray_to_rgb(&img);
    let (mut total_length, mut total_cost) = (0.0, 0.0);
    let start = std::time::Instant::now();
    for i in 0..4 {
        let tree = dijkstra(&cost, &[anchors[i]], true);
        let path = backtrack(&tree, anchors[(i + 1) % 4]);
        total_length += length(&path);
        total_cost += tree.distance[anchors[(i + 1) % 4]];
        draw(&mut rgb, &path, [1.0, 0.5, 0.0]);
        if i == 3 {
            // the tree's distances from the last anchor, as a picture
            save_gray(&tree.distance.map(|d| (d / 150.0).min(1.0)), format!("{dir}/distance-from-seed.png"));
        }
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / 4.0;
    for &a in &anchors {
        rgb[a] = [1.0, 0.0, 0.0];
    }
    save_rgb(&rgb, format!("{dir}/contour-4-anchors.png"));
    println!("closed contour, 4 anchors: length {total_length:.1} px, cost {total_cost:.2}, {ms:.0} ms per Dijkstra run");

    // Only two anchors, on opposite sides of the nucleus: which way does the wire go?
    let mut rgb = gray_to_rgb(&img);
    let tree = dijkstra(&cost, &[anchors[3]], true);
    let path = backtrack(&tree, anchors[1]);
    draw(&mut rgb, &path, [1.0, 0.5, 0.0]);
    let (dx, dy) = (anchors[1].0 as f32 - anchors[3].0 as f32, anchors[1].1 as f32 - anchors[3].1 as f32);
    println!(
        "top to bottom anchor: length {:.1} px (straight line {:.1} px), cost {:.2}",
        length(&path),
        (dx * dx + dy * dy).sqrt(),
        tree.distance[anchors[1]]
    );

    // The same with 4 neighbours and one raster pass: how many pixels does the pass get wrong?
    let tree4 = dijkstra(&cost, &[(0, 0)], false);
    let pass = raster_pass(&cost);
    let wrong = pass.pixels().iter().zip(tree4.distance.pixels()).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
    println!("one raster pass from the top-left corner: {wrong} of {} pixels wrong", pass.pixels().len());
    save_rgb(&rgb, format!("{dir}/wire-2-anchors.png"));
    println!("wrote the results to {dir}/");
}
