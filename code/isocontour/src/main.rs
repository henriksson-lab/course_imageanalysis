// Marching squares on a test image (clean, noisy, blurred) and on the micrograph, with your code;
// contours drawn in out/isocontour/.
//   cargo run -p isocontour                      (your code)
//   cargo run -p isocontour --features solution  (the model solution)

use imgcourse::*;
use isocontour::*;
use std::collections::HashMap;

/// The test image of the demo, 128 × 128, grey values 0 to 255: two smooth blobs, and two discs
/// joined by a neck 3 pixels wide, on a background of 30.
fn test_image() -> Image<f32> {
    Image::from_fn(128, 128, |x, y| {
        let (x, y) = (x as f32, y as f32);
        let blob = |cx: f32, cy: f32, s: f32| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * s * s)).exp();
        let mut v = 30.0 + 190.0 * (blob(36.0, 36.0, 11.0) + blob(70.0, 34.0, 9.0)).min(1.0);
        let disc = |cx: f32, cy: f32, r: f32| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= r;
        let neck = (40.0..=88.0).contains(&x) && (y - 92.0).abs() <= 1.5;
        if disc(40.0, 92.0, 13.0) || disc(88.0, 92.0, 13.0) || neck {
            v = 200.0;
        }
        v
    })
}

/// A Gaussian blur of width `sigma` (pixels), separable, edges copied.
fn blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            (-r..=r).map(|d| {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                w[(d + r) as usize] / s * src.get_border(xx, yy, Border::Clamp)
            }).sum()
        })
    };
    pass(&pass(img, true), false)
}

/// The image inside a frame of one pixel of `value`, so that every contour closes.
fn pad(img: &Image<f32>, value: f32) -> Image<f32> {
    Image::from_fn(img.width() + 2, img.height() + 2, |x, y| {
        if x == 0 || y == 0 || x > img.width() || y > img.height() { value } else { img[(x - 1, y - 1)] }
    })
}

/// Chain the segments into closed contours (end of one = start of the next) and return the
/// signed area of each.
fn contour_areas(segments: &[Segment]) -> Vec<f32> {
    // starts by position, in cells of 0.001; the next segment is the unused one whose start is
    // nearest to this end (the same point, computed by the neighbouring cell, up to rounding)
    let key = |p: Point| ((p.0 * 1000.0).floor() as i64, (p.1 * 1000.0).floor() as i64);
    let mut starts: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, s) in segments.iter().enumerate() {
        starts.entry(key(s.0)).or_default().push(i);
    }
    let mut used = vec![false; segments.len()];
    let mut areas = vec![];
    for first in 0..segments.len() {
        if used[first] {
            continue;
        }
        let mut piece = vec![];
        let mut i = first;
        while !used[i] {
            used[i] = true;
            piece.push(segments[i]);
            let end = segments[i].1;
            let (kx, ky) = key(end);
            let mut best: Option<(f32, usize)> = None;
            for k in (-1..=1).flat_map(|dx| (-1..=1).map(move |dy| (kx + dx, ky + dy))) {
                for &n in starts.get(&k).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let d = (segments[n].0 .0 - end.0).abs() + (segments[n].0 .1 - end.1).abs();
                    if (n == first || !used[n]) && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, n));
                    }
                }
            }
            match best {
                Some((_, n)) if n != first => i = n,
                _ => break,
            }
        }
        areas.push(enclosed_area(&piece));
    }
    areas
}

/// The image (grey, 0–255) enlarged `zoom` times, with the segments drawn in yellow.
fn draw(img: &Image<f32>, segments: &[Segment], zoom: usize) -> Image<[f32; 3]> {
    let mut out = Image::from_fn(img.width() * zoom, img.height() * zoom, |x, y| {
        let v = (img[(x / zoom, y / zoom)] / 255.0).clamp(0.0, 1.0) * 0.8;
        [v, v, v]
    });
    let z = zoom as f32;
    for &((x0, y0), (x1, y1)) in segments {
        let steps = (((x1 - x0).abs() + (y1 - y0).abs()) * z * 2.0) as usize + 1;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            // grid point (x, y) is the centre of pixel (x, y)
            let (px, py) = (((x0 + t * (x1 - x0)) + 0.5) * z, ((y0 + t * (y1 - y0)) + 0.5) * z);
            if out.contains(px as isize, py as isize) {
                out[(px as usize, py as usize)] = [1.0, 0.85, 0.0];
            }
        }
    }
    out
}

fn report(name: &str, img: &Image<f32>, c: f32) -> Vec<Segment> {
    let padded = pad(img, 0.0);
    let segs = marching_squares(&padded, c, true);
    let areas = contour_areas(&segs);
    let holes = areas.iter().filter(|&&a| a < 0.0).count();
    let above = img.pixels().iter().filter(|&&v| v > c).count();
    println!("{name:22} {:5} contours ({:4} outer, {:3} holes)   length {:7.1}   area {:7.1}   pixels above c {above}",
        areas.len(), areas.len() - holes, holes, contour_length(&segs), enclosed_area(&segs));
    segs
}

fn main() {
    let dir = "out/isocontour";
    std::fs::create_dir_all(dir).unwrap();
    let clean = test_image();
    let noisy = Image::from_fn(128, 128, |x, y| clean[(x, y)]);
    let noisy = samples::add_noise(&noisy, 25.0, 1);
    let c = 120.0;
    println!("test image, c = {c} (grey values 0–255), padded with a frame of 0, asymptotic decider:");
    let cases = [
        ("clean", clean.clone()),
        ("noise σ = 25", noisy.clone()),
        ("noise, blur σ = 1.5", blur(&noisy, 1.5)),
        ("noise, blur σ = 2.5", blur(&noisy, 2.5)),
    ];
    for (i, (name, img)) in cases.iter().enumerate() {
        let segs = report(name, img, c);
        // shift by the frame of padding: grid point (x, y) of the padded image is pixel (x − 1, y − 1)
        let shifted: Vec<Segment> = segs.iter().map(|&((a, b), (d, e))| ((a - 1.0, b - 1.0), (d - 1.0, e - 1.0))).collect();
        save_rgb(&draw(img, &shifted, 4), format!("{dir}/contours-{}.png", ["clean", "noise", "blur1.5", "blur2.5"][i]));
    }
    // Two blobs merge as c drops: count the contours of the clean image around the blobs.
    let blobs = clean.crop(0, 0, 128, 64);
    let mut c = 150.0;
    let mut before = 0;
    while c > 30.0 {
        let n = contour_areas(&marching_squares(&pad(&blobs, 0.0), c, true)).len();
        if n != before {
            println!("the two blobs (upper half of the image): from c = {c} down, {n} contour(s)");
            before = n;
        }
        c -= 1.0;
    }
    // The micrograph, blurred a little.
    let cell = blur(&samples::cell().map(|v| 255.0 * v), 1.0);
    let segs = report("micrograph, blur 1", &cell, 100.0);
    let shifted: Vec<Segment> = segs.iter().map(|&((a, b), (d, e))| ((a - 1.0, b - 1.0), (d - 1.0, e - 1.0))).collect();
    save_rgb(&draw(&cell, &shifted, 2), format!("{dir}/contours-cell.png"));
    println!("wrote the pictures to {dir}/");
}
