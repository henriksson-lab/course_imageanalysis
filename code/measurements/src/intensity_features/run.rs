// Measures six synthetic cells with the same mask and the same total signal (as in the
// lecture), prints a feature table, and draws the edge band, centroids and rings to
// out/intensity-features/.
//   cargo run -p measurements -- intensity-features                      (your code)
//   cargo run -p measurements --features solution -- intensity-features  (the model solution)

use imgcourse::*;
use crate::intensity_features::*;

const SIZE: usize = 160;
const RINGS: usize = 4;

/// The mask: an ellipse with semi-axes 52 (across) and 40 (down) in the middle of the image.
fn ellipse(x: usize, y: usize) -> f32 {
    let (u, v) = ((x as f32 - 80.0) / 52.0, (y as f32 - 80.0) / 40.0);
    (u * u + v * v).sqrt() // < 1 inside
}

/// One of the six patterns, blurred a little (the optics) and scaled to mean 0.4 inside the mask.
fn cell(pattern: &str, labels: &Image<u32>) -> Image<f32> {
    let mut rng = Rng::new(3);
    let dots: Vec<(f32, f32)> = (0..25).map(|_| (40.0 + 80.0 * rng.uniform(), 52.0 + 56.0 * rng.uniform())).collect();
    let sharp = Image::from_fn(SIZE, SIZE, |x, y| {
        if labels[(x, y)] == 0 {
            return 0.0;
        }
        let (r, xf, yf) = (ellipse(x, y), x as f32, y as f32);
        match pattern {
            "uniform" => 1.0,
            "bright rim" => if r > 0.85 { 6.0 } else { 0.5 },
            "off-centre spot" => 0.5 + 8.0 * (-((xf - 108.0).powi(2) + (yf - 70.0).powi(2)) / (2.0 * 7.0 * 7.0)).exp(),
            "punctate" => 0.3 + dots.iter().map(|(dx, dy)| 6.0 * (-((xf - dx).powi(2) + (yf - dy).powi(2)) / 4.0).exp()).sum::<f32>(),
            "gradient" => 1.0 + 0.9 * (xf - 80.0) / 52.0,
            _ => if r < 0.5 { 4.0 } else { 0.4 }, // nuclear
        }
    });
    let blurred = blur(&sharp);
    let s = object_stats(&blurred, labels, 1);
    blurred.map(|v| 0.4 * v / s.mean)
}

/// A small binomial blur (weights 1 4 6 4 1, along rows then columns), zeros outside.
fn blur(img: &Image<f32>) -> Image<f32> {
    let w = [1.0, 4.0, 6.0, 4.0, 1.0].map(|v| v / 16.0);
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            (0..5)
                .map(|k| {
                    let d = k as isize - 2;
                    let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                    w[k] * src.get_border(xx, yy, Border::Zero)
                })
                .sum()
        })
    };
    pass(&pass(img, true), false)
}

/// Grow label 1 by one pixel (4-neighbours), `times` times: a mask that is too large.
fn grow(labels: &Image<u32>, times: usize) -> Image<u32> {
    let mut out = labels.clone();
    for _ in 0..times {
        let prev = out.clone();
        out = prev.map_xy(|x, y, l| {
            let near = [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| prev.get_border(x as isize + dx, y as isize + dy, Border::Zero) == 1);
            if near { 1 } else { l }
        });
    }
    out
}

/// The ring (1..=k) of every pixel of object 1, around its geometric centroid; 0 outside.
/// `de` is the distance to the nearest pixel outside the object.
fn rings(labels: &Image<u32>, centre: [f32; 2], k: usize) -> Image<usize> {
    let mut outside = Vec::new(); // pixels just outside the object
    for (x, y) in labels.coords() {
        let (x, y) = (x as isize, y as isize);
        if labels[(x as usize, y as usize)] == 0
            && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| labels.get_border(x + dx, y + dy, Border::Zero) == 1)
        {
            outside.push((x as f32, y as f32));
        }
    }
    labels.map_xy(|x, y, l| {
        if l != 1 {
            return 0;
        }
        let (xf, yf) = (x as f32, y as f32);
        let de = outside.iter().map(|(ox, oy)| ((xf - ox).powi(2) + (yf - oy).powi(2)).sqrt()).fold(f32::INFINITY, f32::min);
        let dc = ((xf - centre[0]).powi(2) + (yf - centre[1]).powi(2)).sqrt();
        ring_of(dc, de, k)
    })
}

pub fn main() {
    let dir = "out/intensity-features";
    let labels = Image::from_fn(SIZE, SIZE, |x, y| if ellipse(x, y) < 1.0 { 1u32 } else { 0 });
    let band = edge_band(&labels, 1);
    let patterns = ["uniform", "bright rim", "off-centre spot", "punctate", "gradient", "nuclear"];

    // One panel per cell, 3 across and 2 down: the cell, the edge band (cyan), the geometric
    // centroid (white) and the intensity-weighted centroid (red). A second sheet: the rings.
    let mut sheet = Image::new(3 * SIZE, 2 * SIZE, [0.0f32; 3]);
    let mut ring_sheet = sheet.clone();
    println!("{} pixels in the mask; values scaled to mean 0.4", object_stats(&labels.map(|_| 1.0), &labels, 1).n);
    println!("{:<16} {:>9} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5}   MeanFrac per ring (centre to edge)", "cell", "integr.", "mean", "sd", "max", "median", "edge/m", "disp");
    for (i, name) in patterns.iter().enumerate() {
        let img = cell(name, &labels);
        let s = object_stats(&img, &labels, 1);
        let edge: Vec<f32> = img.coords().filter(|&p| band[p]).map(|p| img[p]).collect();
        let edge_mean = edge.iter().sum::<f32>() / edge.len() as f32;
        let (g, w, d) = mass_displacement(&img, &labels, 1);
        let r = rings(&labels, g, RINGS);
        let (values, ring_list): (Vec<f32>, Vec<usize>) = img.coords().filter(|&p| r[p] > 0).map(|p| (img[p], r[p])).unzip();
        let (_, mean_frac) = radial_distribution(&values, &ring_list, RINGS);
        let mf: Vec<String> = mean_frac.iter().map(|m| format!("{m:.2}")).collect();
        println!(
            "{:<16} {:>9.1} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.2} {:>5.1}   {}",
            name, s.integrated, s.mean, s.sd, s.max, s.median, edge_mean / s.mean, d, mf.join("  ")
        );

        let (ox, oy) = ((i % 3) * SIZE, (i / 3) * SIZE);
        let top = img.max();
        for (x, y) in img.coords() {
            let v = img[(x, y)] / top;
            sheet[(ox + x, oy + y)] = if band[(x, y)] { [0.0, 0.8, 0.9] } else { [v, v, v] };
            let tint = [[1.0, 0.6, 0.6], [0.6, 1.0, 0.6], [0.6, 0.7, 1.0], [1.0, 0.9, 0.5]];
            ring_sheet[(ox + x, oy + y)] = if r[(x, y)] > 0 { tint[r[(x, y)] - 1].map(|t| t * (0.25 + 0.75 * v)) } else { [0.0; 3] };
        }
        for t in -4..=4isize {
            for (cx, cy, colour) in [(g[0], g[1], [1.0, 1.0, 1.0]), (w[0], w[1], [1.0, 0.1, 0.1])] {
                let (cx, cy) = (cx.round() as isize, cy.round() as isize);
                for (px, py) in [(cx + t, cy), (cx, cy + t)] {
                    sheet[(ox + px as usize, oy + py as usize)] = colour;
                }
            }
        }
    }
    save_rgb(&sheet, format!("{dir}/cells.png"));
    save_rgb(&ring_sheet, format!("{dir}/rings.png"));

    // What survives an offset, and a mask two pixels too large? The uniform cell.
    let img = cell("uniform", &labels);
    let show = |what: &str, s: Stats| {
        println!("{what:<34} integrated {:>7.1}  mean {:.4}  median {:.4}  sd {:.4}  MAD {:.4}", s.integrated, s.mean, s.median, s.sd, s.mad);
    };
    println!();
    show("uniform cell", object_stats(&img, &labels, 1));
    let offset = img.map(|v| v + 0.1); // background 0.1 everywhere
    show("offset 0.1", object_stats(&offset, &labels, 1));
    show("offset 0.1, mask 2 px too large", object_stats(&offset, &grow(&labels, 2), 1));
    println!("wrote {dir}/cells.png and {dir}/rings.png");
}
