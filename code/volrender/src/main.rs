// Renders a synthetic head phantom with your ray sampling and compositing: projections, volume
// rendering and an isosurface. Pictures in out/volrender/.
//   cargo run -p volrender                      (your code)
//   cargo run -p volrender --features solution  (the model solution)

use imgcourse::*;
use volrender::*;

/// Voxel spacing in millimetres: 64 × 64 × 40 voxels of 1 × 1 × 1.6 mm, as in the lecture.
const SPACING: [f32; 3] = [1.0, 1.0, 1.6];

/// The distance from p to the segment a–b.
fn dist_segment(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])).clamp(0.0, 1.0);
    let d = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// A synthetic head (values 0–255): air 0, soft tissue 70, skull 170, brain 100, two
/// ventricles 40, an air cavity 8 below the brain, and a thin vessel tree 230. Positions in mm,
/// y down, the head centred at (32, 32, 32). The edges are soft ramps about 3 mm wide, as a
/// scanner would give.
fn phantom() -> Volume<f32> {
    // 1 well inside (x < edge), 0 well outside, a linear ramp of the given width in between
    let soft = |x: f32, edge: f32, width: f32| ((edge - x) / width + 0.5).clamp(0.0, 1.0);
    let value = |p: [f32; 3]| -> f32 {
        let r = |c: [f32; 3], rad: [f32; 3]| (((p[0] - c[0]) / rad[0]).powi(2) + ((p[1] - c[1]) / rad[1]).powi(2) + ((p[2] - c[2]) / rad[2]).powi(2)).sqrt();
        let head = r([32.0, 32.0, 32.0], [28.0, 30.0, 29.0]);
        let mut v = 70.0 * soft(head, 1.0, 0.1) + 100.0 * soft(head, 0.93, 0.1) - 70.0 * soft(head, 0.82, 0.1);
        v -= 92.0 * soft(r([32.0, 49.0, 24.0], [7.0, 3.5, 6.0]), 1.0, 0.3); // air cavity: 100 → 8
        for c in [[26.0, 28.0, 32.0], [38.0, 28.0, 32.0]] {
            v -= 60.0 * soft(r(c, [3.0, 7.0, 9.0]), 1.0, 0.3); // ventricles: 100 → 40
        }
        let vessels = [
            ([32.0, 60.0, 34.0], [32.0, 38.0, 32.0]),
            ([32.0, 38.0, 32.0], [20.0, 24.0, 26.0]),
            ([32.0, 38.0, 32.0], [44.0, 22.0, 40.0]),
            ([20.0, 24.0, 26.0], [16.0, 14.0, 34.0]),
            ([44.0, 22.0, 40.0], [46.0, 12.0, 30.0]),
        ];
        let d = vessels.iter().map(|&(a, b)| dist_segment(p, a, b)).fold(f32::INFINITY, f32::min);
        v + (230.0 - v) * soft(d, 1.5, 1.5)
    };
    Volume::from_fn(64, 64, 40, |i, j, k| value([i as f32 * SPACING[0], j as f32 * SPACING[1], k as f32 * SPACING[2]]))
}

/// The gradient of the volume at the world point p (central differences of `trilinear`, 3 mm
/// apart), used as the surface normal of an isosurface.
fn gradient(vol: &Volume<f32>, p: [f32; 3]) -> [f32; 3] {
    let at = |q: [f32; 3]| trilinear(vol, q[0] / SPACING[0], q[1] / SPACING[1], q[2] / SPACING[2]);
    [0, 1, 2].map(|k| {
        let (mut a, mut b) = (p, p);
        a[k] += 1.5;
        b[k] -= 1.5;
        at(a) - at(b)
    })
}

/// The transfer function: value → (colour, opacity per millimetre). Air and fluid clear,
/// soft tissue and brain faint, bone a little, vessels nearly opaque and red.
fn transfer(v: f32) -> (Rgb, f32) {
    let ramp = |lo: f32, hi: f32| ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    let alpha = 0.02 * ramp(50.0, 70.0) + 0.4 * ramp(120.0, 160.0) + 0.5 * ramp(195.0, 225.0);
    let colour = if v > 200.0 { [1.0, 0.25, 0.2] } else if v > 140.0 { [1.0, 0.95, 0.85] } else { [0.85, 0.6, 0.45] };
    (colour, alpha.min(1.0))
}

/// One ray per pixel, orthographic, for a camera turned by `angle` degrees about the vertical
/// (y) axis: 128 × 128 pixels of 0.5 mm. Calls `f` with the samples of each ray.
fn render<T: Copy>(vol: &Volume<f32>, angle: f32, step: f32, f: impl Fn(&[f32]) -> T) -> Image<T> {
    let a = angle.to_radians();
    let dir = [a.sin(), 0.0, a.cos()];
    let right = [a.cos(), 0.0, -a.sin()];
    let centre = [32.0, 32.0, 32.0];
    let n = (100.0 / step) as usize;
    Image::from_fn(128, 128, |i, j| {
        let (u, v) = ((i as f32 - 63.5) * 0.5, (j as f32 - 63.5) * 0.5);
        let start = [0, 1, 2].map(|k| centre[k] + u * right[k] + v * [0.0, 1.0, 0.0][k] - 50.0 * dir[k]);
        f(&sample_ray(vol, SPACING, start, dir, step, n))
    })
}

fn main() {
    let dir = "out/volrender";
    let vol = phantom();

    // 1. Projections, from the front (looking along +z) and turned by 90° (from the side).
    for (name, angle) in [("front", 0.0), ("side", 90.0)] {
        let mip = render(&vol, angle, 0.5, |s| s.iter().cloned().fold(0.0f32, f32::max));
        save_gray_stretched(&mip, format!("{dir}/mip-{name}.png"));
    }
    let avg = render(&vol, 0.0, 0.5, |s| s.iter().sum::<f32>() / s.len() as f32);
    save_gray_stretched(&avg, format!("{dir}/average-front.png"));

    // 2. Volume rendering at 30°: transfer function, opacity corrected to the step, composited
    //    front to back with early termination at 0.05, over a black background.
    let dvr = |step: f32, correct: bool, eps: f32| {
        render(&vol, 30.0, step, move |s| {
            let samples: Vec<(Rgb, f32)> = s.iter().map(|&v| {
                let (c, a) = transfer(v);
                (c, if correct { opacity_correct(a, step) } else { a })
            }).collect();
            front_to_back(&samples, eps)
        })
    };
    for (name, step, correct) in [("step0.5", 0.5, true), ("step3", 3.0, true), ("step0.25-uncorrected", 0.25, false), ("step0.25", 0.25, true), ("step1", 1.0, true)] {
        let out = dvr(step, correct, 0.05);
        let opacity: f32 = out.pixels().iter().map(|p| 1.0 - p.1).sum::<f32>() / (128.0 * 128.0);
        println!("volume rendering, {name:21}: mean opacity of the rays {opacity:.3}");
        save_rgb(&out.map(|p| p.0), format!("{dir}/dvr-{name}.png"));
    }

    // 3. Early termination: how many samples the rays take, with and without it.
    let without = dvr(0.5, true, 0.0);
    let with = dvr(0.5, true, 0.05);
    let total = |img: &Image<(Rgb, f32, usize)>| img.pixels().iter().map(|p| p.2).sum::<usize>() as f32 / (128.0 * 128.0);
    let stopped = with.pixels().iter().filter(|p| p.2 < 200).count();
    println!("early termination: {:.1} samples per ray instead of {:.1}; {stopped} of {} rays stop early", total(&with), total(&without), 128 * 128);
    let fewest = with.pixels().iter().map(|p| p.2).min().unwrap();
    println!("  fewest samples on one ray: {fewest}; the ray through the middle of the head, pixel (80, 64): {}", with[(80, 64)].2);
    save_gray_stretched(&with.map(|p| p.2 as f32), format!("{dir}/samples-per-ray.png"));

    // 4. Isosurfaces by ray casting at 30°: the first crossing of c, lit from the viewer's
    //    direction with the gradient as the normal (Lambert).
    let a = 30.0f32.to_radians();
    let view = [a.sin(), 0.0, a.cos()];
    for c in [60.0, 135.0, 200.0] {
        let iso = Image::from_fn(128, 128, |i, j| {
            let (u, v) = ((i as f32 - 63.5) * 0.5, (j as f32 - 63.5) * 0.5);
            let start = [32.0 + u * a.cos() - 50.0 * view[0], 32.0 + v, 32.0 - u * a.sin() - 50.0 * view[2]];
            let s = sample_ray(&vol, SPACING, start, view, 0.5, 200);
            match iso_crossing(&s, c) {
                Some(t) => {
                    let p = [0, 1, 2].map(|k| start[k] + t * 0.5 * view[k]);
                    let g = gradient(&vol, p);
                    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-6);
                    // the gradient points inwards (values rise), towards +view on the near side
                    0.15 + 0.85 * ((g[0] * view[0] + g[1] * view[1] + g[2] * view[2]) / len).max(0.0)
                }
                None => 0.0,
            }
        });
        save_gray(&iso, format!("{dir}/iso-{c}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
