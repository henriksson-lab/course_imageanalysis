// Distance transforms, medial axes, Lantuéjoul skeletons and reconstructions of a few shapes
// (a rectangle, a star, an ellipse with and without a bump); counts printed, pictures in
// out/skel-medialaxis/.
//   cargo run -p day17 -- skel-medialaxis                      (your code)
//   cargo run -p day17 --features solution -- skel-medialaxis  (the model solution)

use imgcourse::*;
use crate::skel_medialaxis::*;

const SIZE: usize = 256;

/// Is the point inside the polygon? (Count crossings of a ray to the right.)
fn in_polygon(px: f32, py: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    for i in 0..poly.len() {
        let ((x0, y0), (x1, y1)) = (poly[i], poly[(i + 1) % poly.len()]);
        if (y0 > py) != (y1 > py) && px < x0 + (py - y0) / (y1 - y0) * (x1 - x0) {
            inside = !inside;
        }
    }
    inside
}

fn shapes() -> Vec<(&'static str, Image<bool>)> {
    let rect = Image::from_fn(SIZE, SIZE, |x, y| (40..216).contains(&x) && (92..164).contains(&y));
    let star: Vec<(f32, f32)> = (0..10)
        .map(|i| {
            let a = std::f32::consts::PI * (i as f32 / 5.0 - 0.5);
            let r = if i % 2 == 0 { 110.0 } else { 45.0 };
            (128.0 + r * a.cos(), 132.0 + r * a.sin())
        })
        .collect();
    let star = Image::from_fn(SIZE, SIZE, |x, y| in_polygon(x as f32, y as f32, &star));
    let ellipse = |x: usize, y: usize| ((x as f32 - 128.0) / 100.0).powi(2) + ((y as f32 - 128.0) / 50.0).powi(2) <= 1.0;
    let plain = Image::from_fn(SIZE, SIZE, ellipse);
    // a bump of radius 3 on the lower boundary, right of the middle
    let bump = Image::from_fn(SIZE, SIZE, |x, y| ellipse(x, y) || (x as f32 - 160.0).powi(2) + (y as f32 - 175.5).powi(2) <= 9.0);
    vec![("rectangle", rect), ("star", star), ("ellipse", plain), ("ellipse-bump", bump)]
}

/// The number of 8-connected pieces of a binary image.
fn pieces(img: &Image<bool>) -> usize {
    let mut seen = img.map(|_| false);
    let mut count = 0;
    for start in img.coords() {
        if !img[start] || seen[start] {
            continue;
        }
        count += 1;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some((x, y)) = stack.pop() {
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if img.contains(nx, ny) && img[(nx as usize, ny as usize)] && !seen[(nx as usize, ny as usize)] {
                        seen[(nx as usize, ny as usize)] = true;
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
    }
    count
}

fn count(img: &Image<bool>) -> usize {
    img.pixels().iter().filter(|&&v| v).count()
}

/// The distance transform in grey, the skeleton in red.
fn picture(dt: &Image<u32>, skel: &Image<bool>) -> Image<[f32; 3]> {
    let top = dt.pixels().iter().copied().max().unwrap_or(1).max(1) as f32;
    dt.map_xy(|x, y, d| if skel[(x, y)] { [1.0, 0.15, 0.1] } else { [d as f32 / top; 3].map(|v| 0.15 + 0.85 * v * (d > 0) as u8 as f32) })
}

pub fn main() {
    let dir = "out/skel-medialaxis";
    println!("{:<13} {:<10} {:>6} {:>7} {:>9} {:>6}", "shape", "metric", "max DT", "axis px", "pieces", "missed");
    for (name, mask) in shapes() {
        for (mname, metric) in [("city-block", Metric::CityBlock), ("chessboard", Metric::Chessboard)] {
            let dt = distance_transform(&mask, metric);
            let axis = medial_axis(&dt, metric);
            let back = reconstruct(&axis, &dt, metric);
            let missed = mask.zip_map(&back, |a, b| a != b);
            let max = dt.pixels().iter().copied().max().unwrap_or(0);
            println!("{:<13} {:<10} {:>6} {:>7} {:>9} {:>6}", name, mname, max, count(&axis), pieces(&axis), count(&missed));
            save_rgb(&picture(&dt, &axis), format!("{dir}/{name}-{mname}.png"));
        }
        let lant = lantuejoul(&mask, false);
        let skel = lant.map(|v| v > 0);
        let back = reconstruct_lantuejoul(&lant, false);
        let missed = mask.zip_map(&back, |a, b| a != b);
        println!(
            "{:<13} {:<10} {:>6} {:>7} {:>9} {:>6}",
            name,
            "Lantuéjoul",
            lant.pixels().iter().copied().max().unwrap_or(0),
            count(&skel),
            pieces(&skel),
            count(&missed)
        );
        save_rgb(&picture(&distance_transform(&mask, Metric::Chessboard), &skel), format!("{dir}/{name}-lantuejoul.png"));
    }
    println!("(max DT for Lantuéjoul: the largest r + 1; missed = pixels where the reconstruction differs from the shape)");
    println!("wrote the pictures to {dir}/ (grey: distance transform, red: skeleton)");
}
