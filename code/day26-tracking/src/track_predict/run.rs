// Follows an object along a line with a random-walk and a constant-velocity Kalman filter:
// it moves at 1 unit per frame, turns back at frame 20 (velocity −0.8), and is missed in
// frames 12 to 14. Prints a table and draws out/track-predict/filters.png.
//   cargo run -p day26 -- track-predict                      (your code)
//   cargo run -p day26 --features solution -- track-predict  (the model solution)

use imgcourse::*;
use crate::track_predict::*;

const FRAMES: usize = 40;
const R: f32 = 4.0; // measurement variance (standard deviation 2)

/// Plot coordinates: frame `t` and position `v` to a pixel of a 640 × 400 picture.
fn to_pixel(t: f32, v: f32) -> (isize, isize) {
    ((20.0 + t * 15.5) as isize, (380.0 - (v + 8.0) * 10.0) as isize)
}

fn dot(img: &mut Image<[f32; 3]>, t: f32, v: f32, r: isize, colour: [f32; 3]) {
    let (x, y) = to_pixel(t, v);
    for dy in -r..=r {
        for dx in -r..=r {
            if img.contains(x + dx, y + dy) {
                img[((x + dx) as usize, (y + dy) as usize)] = colour;
            }
        }
    }
}

fn polyline(img: &mut Image<[f32; 3]>, values: &[f32], colour: [f32; 3]) {
    for t in 0..values.len() - 1 {
        for k in 0..=20 {
            let s = k as f32 / 20.0;
            dot(img, t as f32 + s, values[t] + s * (values[t + 1] - values[t]), 1, colour);
        }
    }
}

pub fn main() {
    let dir = "out/track-predict";
    // The truth and the measurements (seeded noise); frames 12, 13, 14 are missed.
    let mut rng = Rng::new(5);
    let mut truth = vec![0.0f32];
    for t in 1..FRAMES {
        let v = if t <= 20 { 1.0 } else { -0.8 };
        truth.push(truth[t - 1] + v);
    }
    let zs: Vec<Option<f32>> = (0..FRAMES)
        .map(|t| if (12..=14).contains(&t) { None } else { Some(truth[t] + R.sqrt() * rng.normal()) })
        .collect();

    // Random walk (Q = 0.1) and constant velocity (velocity noise 0.01), both starting at
    // the first measurement with variance R.
    let z0 = zs[0].unwrap();
    let mut rw = vec![Estimate { x: z0, p: R }];
    rw.extend(run_filter(rw[0], &zs[1..], 0.1, R));
    let mut cv = vec![CvEstimate { x: [z0, 0.0], p: [[R, 0.0], [0.0, 1.0]] }];
    for z in &zs[1..] {
        let mut e = cv_predict(*cv.last().unwrap(), 1.0, 0.01);
        if let Some(z) = z {
            e = cv_update(e, *z, R).0;
        }
        cv.push(e);
    }

    println!("frame  truth  measured   random walk (x, P)   constant velocity (x, v, P)");
    for t in [1, 5, 10, 11, 12, 13, 14, 15, 19, 20, 22, 25, 29, 35, 39] {
        let z = zs[t].map_or("missed".to_string(), |z| format!("{z:6.2}"));
        println!(
            "{t:5}  {:5.1}   {z:>6}    {:6.2}  {:5.2}        {:6.2}  {:5.2}  {:5.2}",
            truth[t], rw[t].x, rw[t].p, cv[t].x[0], cv[t].x[1], cv[t].p[0][0]
        );
    }
    let worst = |f: &dyn Fn(usize) -> f32| (21..FRAMES).map(|t| (f(t) - truth[t]).abs()).fold(0.0f32, f32::max);
    println!("largest error after the turn: random walk {:.2}, constant velocity {:.2}", worst(&|t| rw[t].x), worst(&|t| cv[t].x[0]));

    // The picture: ±2σ band of the constant-velocity filter (light blue), truth (grey),
    // measurements (orange), random walk (green), constant velocity (dark blue).
    let mut img = Image::new(640, 400, [1.0f32; 3]);
    for t in 0..FRAMES {
        let sd = cv[t].p[0][0].sqrt();
        for k in 0..=40 {
            let v = cv[t].x[0] - 2.0 * sd + 4.0 * sd * k as f32 / 40.0;
            for dt in 0..8 {
                dot(&mut img, t as f32 + dt as f32 / 16.0 - 0.25, v, 1, [0.82, 0.9, 1.0]);
            }
        }
    }
    polyline(&mut img, &truth, [0.6, 0.6, 0.6]);
    polyline(&mut img, &rw.iter().map(|e| e.x).collect::<Vec<_>>(), [0.1, 0.65, 0.2]);
    polyline(&mut img, &cv.iter().map(|e| e.x[0]).collect::<Vec<_>>(), [0.1, 0.2, 0.7]);
    for (t, z) in zs.iter().enumerate() {
        match z {
            Some(z) => dot(&mut img, t as f32, *z, 3, [1.0, 0.5, 0.0]),
            None => dot(&mut img, t as f32, truth[t], 3, [0.9, 0.0, 0.0]), // where the miss was
        }
    }
    save_rgb(&img, format!("{dir}/filters.png"));
    println!("wrote {dir}/filters.png: truth grey, measurements orange (missed: red), random walk green,");
    println!("constant velocity dark blue with ±2σ bars light blue; frames across, position up");
}
