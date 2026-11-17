// Measures the shape of thresholded nuclei and writes the results to out/shape-features/.
//   cargo run -p day14 -- shape-features                      (your code)
//   cargo run -p day14 --features solution -- shape-features  (the model solution)

use imgcourse::*;
use crate::shape_features::*;
use std::collections::{BTreeSet, VecDeque};

/// Label the 8-connected components of a mask (the labelling unit's flood fill).
fn label8(mask: &Image<bool>) -> (Image<u32>, u32) {
    let mut labels = Image::new(mask.width(), mask.height(), 0u32);
    let mut n = 0;
    for start in mask.coords() {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n;
        let mut queue = VecDeque::from([start]);
        while let Some((x, y)) = queue.pop_front() {
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if mask.contains(nx, ny) && mask[(nx as usize, ny as usize)] && labels[(nx as usize, ny as usize)] == 0 {
                        labels[(nx as usize, ny as usize)] = n;
                        queue.push_back((nx as usize, ny as usize));
                    }
                }
            }
        }
    }
    (labels, n)
}

/// Draw a straight line on a colour image (pixel (x, y) is drawn at its centre, x + ½).
fn line(img: &mut Image<[f32; 3]>, a: Point, b: Point, colour: [f32; 3]) {
    let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) * 2.0).ceil() as usize + 1;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let (x, y) = (a.0 + t * (b.0 - a.0) - 0.5, a.1 + t * (b.1 - a.1) - 0.5);
        let (xi, yi) = (x.round() as isize, y.round() as isize);
        if img.contains(xi, yi) {
            img[(xi as usize, yi as usize)] = colour;
        }
    }
}

pub fn main() {
    let dir = "out/shape-features";
    let (img, truth) = samples::nuclei(256, 30, 7);
    let mask = img.map(|v| v > 0.35);
    let (labels, n) = label8(&mask);
    save_gray(&img, format!("{dir}/nuclei.png"));

    let mut picture = gray_to_rgb(&mask.map(|m| if m { 0.35 } else { 0.0 }));
    println!("object  area  hull area  solidity  min Feret  max Feret  inscribed r  nuclei inside");
    for k in 1..=n {
        let pixels: Vec<(usize, usize)> = labels.coords().filter(|&p| labels[p] == k).collect();
        // the object alone, cut out with one pixel of background around it
        let x0 = pixels.iter().map(|p| p.0).min().unwrap().saturating_sub(1);
        let y0 = pixels.iter().map(|p| p.1).min().unwrap().saturating_sub(1);
        let x1 = (pixels.iter().map(|p| p.0).max().unwrap() + 2).min(256);
        let y1 = (pixels.iter().map(|p| p.1).max().unwrap() + 2).min(256);
        let only = Image::from_fn(x1 - x0, y1 - y0, |x, y| labels[(x0 + x, y0 + y)] == k);

        let hull = convex_hull(&hull_candidates(&only));
        let hull_area = polygon_area(&hull);
        let solidity = pixels.len() as f32 / hull_area;
        let (fmin, fmax) = feret(&hull);
        let inscribed = distance_transform(&only).max() - 0.5;
        // how many true nuclei have at least 20 pixels in this object?
        let mut count = std::collections::HashMap::new();
        for &p in &pixels {
            if truth[p] > 0 {
                *count.entry(truth[p]).or_insert(0) += 1;
            }
        }
        let inside: BTreeSet<u32> = count.iter().filter(|(_, &c)| c >= 20).map(|(&l, _)| l).collect();
        println!("{k:6} {:5} {hull_area:10.1} {solidity:9.3} {fmin:10.1} {fmax:10.1} {inscribed:12.1} {:14}", pixels.len(), inside.len());

        // hull in green, or red if the solidity is below 0.9
        let colour = if solidity < 0.9 { [1.0, 0.2, 0.2] } else { [0.3, 1.0, 0.3] };
        for i in 0..hull.len() {
            let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
            line(&mut picture, (a.0 + x0 as f32, a.1 + y0 as f32), (b.0 + x0 as f32, b.1 + y0 as f32), colour);
        }
    }
    save_rgb(&picture, format!("{dir}/hulls.png"));
    println!("wrote the results to {dir}/");
}
