// Runs snakes with your explicit and semi-implicit steps and your gradient vector flow;
// pictures in out/snake-matrix/.
//   cargo run -p snake-matrix                      (your code)
//   cargo run -p snake-matrix --features solution  (the model solution)

use imgcourse::*;
use snake_matrix::*;

type Point = [f32; 2];

/// The settings of one snake run.
struct Run {
    alpha: f64,
    beta: f64,
    w: f32,
    tau: f64,
    semi_implicit: bool,
    steps: usize,
    resample_every: usize,
}

/// Move the snake `steps` times in the force field `(fx, fy)` (read between pixels).
fn run(start: &[Point], fx: &Image<f32>, fy: &Image<f32>, r: &Run) -> Vec<Point> {
    let mut xs: Vec<f64> = start.iter().map(|p| p[0] as f64).collect();
    let mut ys: Vec<f64> = start.iter().map(|p| p[1] as f64).collect();
    // The matrix depends only on n, α, β and τ: build and invert it once.
    let a = snake_matrix(xs.len(), r.alpha, r.beta);
    let inv = semi_implicit_inverse(&a, r.tau);
    for step in 1..=r.steps {
        let force = |field: &Image<f32>| -> Vec<f64> {
            xs.iter().zip(&ys).map(|(&x, &y)| (r.w * bilinear(field, x as f32, y as f32)) as f64).collect()
        };
        let (gx, gy) = (force(fx), force(fy));
        if r.semi_implicit {
            xs = semi_implicit_step(&inv, &xs, &gx, r.tau);
            ys = semi_implicit_step(&inv, &ys, &gy, r.tau);
        } else {
            xs = explicit_step(&a, &xs, &gx, r.tau);
            ys = explicit_step(&a, &ys, &gy, r.tau);
        }
        if xs.iter().chain(&ys).any(|v| v.abs() > 1e4) {
            println!("    exploded after {step} steps");
            break;
        }
        if r.resample_every > 0 && step % r.resample_every == 0 {
            let p = resample(&xs.iter().zip(&ys).map(|(&x, &y)| [x as f32, y as f32]).collect::<Vec<_>>(), xs.len());
            xs = p.iter().map(|q| q[0] as f64).collect();
            ys = p.iter().map(|q| q[1] as f64).collect();
        }
    }
    xs.iter().zip(&ys).map(|(&x, &y)| [x as f32, y as f32]).collect()
}

/// `n` points spaced evenly along the closed polygon.
fn resample(p: &[Point], n: usize) -> Vec<Point> {
    let m = p.len();
    let seg: Vec<f32> = (0..m).map(|i| dist(p[i], p[(i + 1) % m])).collect();
    let total: f32 = seg.iter().sum();
    let mut out = Vec::with_capacity(n);
    let (mut i, mut walked) = (0, 0.0);
    for k in 0..n {
        let target = total * k as f32 / n as f32;
        while walked + seg[i] < target && i < m - 1 {
            walked += seg[i];
            i += 1;
        }
        let t = if seg[i] > 0.0 { (target - walked) / seg[i] } else { 0.0 };
        let (a, b) = (p[i], p[(i + 1) % m]);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    out
}

/// The field scaled to arrows of length 1 (GVF is weak far from edges, but points the right way).
fn unit_length((u, v): &(Image<f32>, Image<f32>)) -> (Image<f32>, Image<f32>) {
    let len = u.zip_map(v, |a, b| (a * a + b * b).sqrt().max(1e-12));
    (u.zip_map(&len, |a, l| a / l), v.zip_map(&len, |b, l| b / l))
}

fn dist(a: Point, b: Point) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn circle(cx: f32, cy: f32, r: f32, n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            [cx + r * t.cos(), cy + r * t.sin()]
        })
        .collect()
}

fn mean_radius(p: &[Point], c: Point) -> f32 {
    p.iter().map(|&q| dist(q, c)).sum::<f32>() / p.len() as f32
}

fn line(img: &mut Image<[f32; 3]>, a: Point, b: Point, colour: [f32; 3]) {
    let steps = (dist(a, b) * 2.0).ceil().max(1.0) as usize;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let (x, y) = ((a[0] + t * (b[0] - a[0])).round() as isize, (a[1] + t * (b[1] - a[1])).round() as isize);
        if img.contains(x, y) {
            img[(x as usize, y as usize)] = colour;
        }
    }
}

fn polygon(img: &mut Image<[f32; 3]>, p: &[Point], colour: [f32; 3]) {
    for i in 0..p.len() {
        line(img, p[i], p[(i + 1) % p.len()], colour);
    }
}

/// The force field as short green arrows on a grid, scaled so that the longest is `len` pixels.
fn arrows(img: &mut Image<[f32; 3]>, fx: &Image<f32>, fy: &Image<f32>, every: usize, len: f32) {
    let m = fx.zip_map(fy, |a, b| (a * a + b * b).sqrt()).max().max(1e-12);
    for y in (every / 2..fx.height()).step_by(every) {
        for x in (every / 2..fx.width()).step_by(every) {
            let (u, v) = (fx[(x, y)] / m * len, fy[(x, y)] / m * len);
            line(img, [x as f32, y as f32], [x as f32 + u, y as f32 + v], [0.1, 0.8, 0.2]);
        }
    }
}

/// Side by side, enlarged twice.
fn side_by_side(a: &Image<[f32; 3]>, b: &Image<[f32; 3]>) -> Image<[f32; 3]> {
    let (w, h) = (a.width(), a.height());
    Image::from_fn(2 * (2 * w + 4), 2 * h, |x, y| {
        let (x, y) = (x / 2, y / 2);
        if x < w {
            a[(x, y)]
        } else if x < w + 4 {
            [1.0; 3]
        } else {
            b[(x - w - 4, y)]
        }
    })
}

fn main() {
    let dir = "out/snake-matrix";
    let (yellow, white) = ([1.0, 0.9, 0.0], [1.0, 1.0, 1.0]);
    let c = [64.0, 64.0];

    // Without an image: how fast does the overall size (mode 1) shrink? (B5)
    let flat = Image::new(128, 128, 0.0f32);
    let unit1 = Run { alpha: 0.1, beta: 0.05, w: 0.0, tau: 1.0, semi_implicit: true, steps: 278, resample_every: 0 };
    let s = run(&circle(64.0, 64.0, 40.0, 40), &flat, &flat, &unit1);
    println!("no image, radius 40, 278 semi-implicit steps (tau = 1): radius {:.2}", mean_radius(&s, c));

    // Unit 1's disk: the explicit step with tau = 2 explodes, the semi-implicit one does not.
    let disk = Image::from_fn(128, 128, |x, y| if dist([x as f32, y as f32], c) < 30.0 { 0.8 } else { 0.2 });
    let disk = samples::add_noise(&disk, 0.05, 1);
    let f = edge_map(&disk, 2.0);
    let (fx, fy) = gradient(&f);
    let start = circle(64.0, 64.0, 45.0, 40);
    for semi_implicit in [false, true] {
        let name = if semi_implicit { "semi-implicit" } else { "explicit" };
        println!("disk, start radius 45, tau = 2, {name}:");
        let s = run(&start, &fx, &fy, &Run { tau: 2.0, semi_implicit, steps: 300, w: 2.0, ..unit1 });
        if s.iter().all(|p| p[0].abs() < 1e4) {
            println!("    after 300 steps: mean radius {:.2}", mean_radius(&s, c));
        }
    }

    // Capture range: a far start (radius 58, alpha = 0.01), edge force against GVF.
    let (gx, gy) = unit_length(&gvf(&fx, &fy, 0.2, 1.0, 1000));
    let far = circle(64.0, 64.0, 58.0, 40);
    let mut pics = Vec::new();
    for (name, ux, uy) in [("edge force", &fx, &fy), ("GVF", &gx, &gy)] {
        let s = run(&far, ux, uy, &Run { alpha: 0.01, w: 2.0, steps: 100, ..unit1 });
        println!("far start (radius 58), {name}, 100 steps: mean radius {:.2}", mean_radius(&s, c));
        let mut pic = gray_to_rgb(&disk);
        arrows(&mut pic, ux, uy, 8, 6.0);
        polygon(&mut pic, &far, white);
        polygon(&mut pic, &s, yellow);
        pics.push(pic);
    }
    save_rgb(&side_by_side(&pics[0], &pics[1]), format!("{dir}/far-start.png"));

    // A U-shaped object: can the snake enter the slot?
    let u = Image::from_fn(128, 128, |x, y| {
        let (x, y) = (x as f32, y as f32);
        let outer = (24.0..104.0).contains(&x) && (14.0..104.0).contains(&y);
        let slot = (50.0..78.0).contains(&x) && y < 78.0;
        if outer && !slot { 0.8 } else { 0.2 }
    });
    let u = samples::add_noise(&u, 0.05, 2);
    let f = edge_map(&u, 2.0);
    let (fx, fy) = gradient(&f);
    let (gx, gy) = unit_length(&gvf(&fx, &fy, 0.2, 1.0, 1000));
    let start = circle(64.0, 60.0, 58.0, 100);
    let mut pics = Vec::new();
    for (name, ux, uy) in [("edge force", &fx, &fy), ("GVF", &gx, &gy)] {
        let s = run(&start, ux, uy, &Run { alpha: 0.03, beta: 0.01, w: 2.0, steps: 800, resample_every: 10, ..unit1 });
        let deepest = s.iter().filter(|p| (52.0..76.0).contains(&p[0]) && p[1] < 95.0).map(|p| p[1]).fold(0.0, f32::max);
        println!("U shape, {name}, 800 steps: deepest point in the slot at y = {deepest:.1} (the slot ends at y = 77.5)");
        let mut pic = gray_to_rgb(&u);
        arrows(&mut pic, ux, uy, 8, 6.0);
        polygon(&mut pic, &s, yellow);
        pics.push(pic);
    }
    save_rgb(&side_by_side(&pics[0], &pics[1]), format!("{dir}/u-shape.png"));
    println!("wrote the pictures to {dir}/");
}
