// Measures objects and perimeters, and writes the results to out/regionprops/.
//   cargo run -p regionprops                      (your code)
//   cargo run -p regionprops --features solution  (the model solution)

use imgcourse::*;
use regionprops::*;
use std::f32::consts::PI;

/// Set a pixel of a colour image if it is inside.
fn plot(img: &mut Image<[f32; 3]>, x: f32, y: f32, colour: [f32; 3]) {
    let (xi, yi) = (x.round() as isize, y.round() as isize);
    if img.contains(xi, yi) {
        img[(xi as usize, yi as usize)] = colour;
    }
}

fn main() {
    let dir = "out/regionprops";

    // 1. The measurement table of the true labels of a synthetic nuclei image, and the
    //    equivalent ellipses drawn over the image.
    let (img, labels) = samples::nuclei(256, 12, 5);
    save_gray(&img, format!("{dir}/nuclei.png"));
    save_labels(&labels, format!("{dir}/nuclei-labels.png"));
    let regions = measure(&labels);
    let mut picture = gray_to_rgb(&img);
    println!("label  area      x̄      ȳ     λ1     λ2  angle  eccentr.  extent  edges  Crofton  4πA/P²");
    for (k, r) in regions.iter().enumerate() {
        if r.area == 0 {
            continue;
        }
        let e = ellipse(r);
        let only = labels.map(|l| l == k as u32 + 1);
        let edges = perimeter_edges(&only);
        let crofton = perimeter_crofton(&only);
        println!(
            "{:5} {:5} {:6.1} {:6.1} {:6.1} {:6.1} {:5.0}° {:9.2} {:7.2} {:6} {:8.1} {:7.2}",
            k + 1,
            r.area,
            r.cx,
            r.cy,
            e.lambda1,
            e.lambda2,
            -e.theta.to_degrees(), // counter-clockwise on screen, as in the lecture
            e.eccentricity(),
            r.extent(),
            edges,
            crofton,
            4.0 * PI * r.area as f32 / (crofton * crofton)
        );
        // the equivalent ellipse (orange) and its major axis (red)
        let (a, b) = e.semi_axes();
        let (c, s) = (e.theta.cos(), e.theta.sin());
        for i in 0..720 {
            let t = i as f32 * PI / 360.0;
            let (u, v) = (a * t.cos(), b * t.sin());
            plot(&mut picture, r.cx + u * c - v * s, r.cy + u * s + v * c, [1.0, 0.6, 0.0]);
        }
        for i in -40..=40 {
            let u = a * i as f32 / 40.0;
            plot(&mut picture, r.cx + u * c, r.cy + u * s, [1.0, 0.1, 0.1]);
        }
    }
    save_rgb(&picture, format!("{dir}/nuclei-ellipses.png"));

    // 2. Perimeter estimators on digital discs of growing radius, against the circle of the
    //    same area.
    println!("\nradius   area   edges/P  Crofton/P   (P = perimeter of the circle with the same area)");
    for r in [2.0f32, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0] {
        let n = 2 * r as usize + 5;
        let c = (n / 2) as f32;
        let d = Image::from_fn(n, n, |x, y| (x as f32 - c).powi(2) + (y as f32 - c).powi(2) <= r * r);
        let area = d.pixels().iter().filter(|&&v| v).count();
        let p = 2.0 * (PI * area as f32).sqrt();
        println!("{r:6} {area:6} {:9.3} {:10.3}", perimeter_edges(&d) as f32 / p, perimeter_crofton(&d) / p);
    }

    // 3. Euler numbers of a whole mask, without labelling (grey level below 95 of 255).
    let dark = samples::cell().map(|v| v < 95.0 / 255.0);
    save_gray(&dark.map(|m| m as u8 as f32), format!("{dir}/cell-dark.png"));
    println!(
        "\nEuler number of the dark regions of the micrograph: {} (objects 8), {} (objects 4)",
        euler_number(&dark, true),
        euler_number(&dark, false)
    );
    println!("wrote the results to {dir}/");
}
