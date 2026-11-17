// Links a simulated movie of randomly moving particles with nearest neighbour, greedy and
// global assignment, at three difficulty ratios; pictures in out/track-linking/.
//   cargo run -p day26 -- track-linking                      (your code)
//   cargo run -p day26 --features solution -- track-linking  (the model solution)

use imgcourse::*;
use crate::track_linking::*;

const SIZE: f32 = 256.0; // the field is SIZE × SIZE pixels
const PARTICLES: usize = 80;
const FRAMES: usize = 40;

/// `movie[t][i]` is particle `i` at frame `t`: a random walk with `step_sd` per axis and
/// frame, bouncing off the walls. The index is the true identity.
fn simulate(step_sd: f32, seed: u64) -> Vec<Vec<Point>> {
    let mut rng = Rng::new(seed);
    let mut pos: Vec<Point> = (0..PARTICLES).map(|_| [SIZE * rng.uniform(), SIZE * rng.uniform()]).collect();
    let mut movie = vec![pos.clone()];
    for _ in 1..FRAMES {
        for p in pos.iter_mut() {
            for v in p.iter_mut() {
                *v += step_sd * rng.normal();
                if *v < 0.0 {
                    *v = -*v;
                }
                if *v > SIZE {
                    *v = 2.0 * SIZE - *v;
                }
            }
        }
        movie.push(pos.clone());
    }
    movie
}

/// Draw a line from `a` to `b` (in field coordinates) on a picture `scale` times larger.
fn line(img: &mut Image<[f32; 3]>, a: Point, b: Point, scale: f32, colour: [f32; 3]) {
    let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * scale).ceil().max(1.0) as usize;
    for k in 0..=steps {
        let t = k as f32 / steps as f32;
        let (x, y) = ((a[0] + t * (b[0] - a[0])) * scale, (a[1] + t * (b[1] - a[1])) * scale);
        if img.contains(x as isize, y as isize) {
            img[(x as usize, y as usize)] = colour;
        }
    }
}

pub fn main() {
    let dir = "out/track-linking";
    let spacing = 0.5 * SIZE / (PARTICLES as f32).sqrt(); // mean nearest-neighbour spacing
    println!("{PARTICLES} particles in {SIZE} × {SIZE} px, {FRAMES} frames, spacing {spacing:.1} px");
    println!("ratio  a/(1+a)   nearest  greedy  global   perfect tracks (global)  (1 - e)^{}", FRAMES - 1);

    for ratio in [0.2f32, 0.5, 1.0] {
        let rms = ratio * spacing; // rms displacement per frame (2D)
        let movie = simulate(rms / 2f32.sqrt(), 1);
        let gate = 5.0 * rms;
        let b = (3.0 * rms).powi(2); // a link is taken if its cost < 2b
        let a = std::f32::consts::PI / 4.0 * ratio * ratio;

        let mut wrong = [0usize; 3];
        let mut perfect = [true; PARTICLES];
        let mut picture = [Image::new(512, 512, [0.0f32; 3]), Image::new(512, 512, [0.0f32; 3])];
        for t in 0..FRAMES - 1 {
            let cost = cost_matrix(&movie[t], &movie[t + 1], gate);
            let results = [nearest(&cost), greedy(&cost), link_with_ends(&cost, b, hungarian)];
            for (m, links) in results.iter().enumerate() {
                for (i, link) in links.iter().enumerate() {
                    let right = *link == Some(i); // the true successor of i is i
                    if !right {
                        wrong[m] += 1;
                        if m == 2 {
                            perfect[i] = false;
                        }
                    }
                    // pictures of the first 15 frames at ratio 0.5: nearest and global
                    if ratio == 0.5 && t < 15 && (m == 0 || m == 2) {
                        if let Some(j) = link {
                            let colour = if right { [0.2, 0.9, 0.3] } else { [1.0, 0.15, 0.15] };
                            line(&mut picture[m / 2], movie[t][i], movie[t + 1][*j], 2.0, colour);
                        }
                    }
                }
            }
        }
        let links = (PARTICLES * (FRAMES - 1)) as f32;
        let e = wrong.map(|w| 100.0 * w as f32 / links);
        let n_perfect = perfect.iter().filter(|&&p| p).count();
        println!(
            "{ratio:5.1}  {:6.1} %  {:6.1} % {:6.1} % {:6.1} %   {:3} of {PARTICLES} ({:.0} %)          {:.0} %",
            100.0 * a / (1.0 + a),
            e[0],
            e[1],
            e[2],
            n_perfect,
            100.0 * n_perfect as f32 / PARTICLES as f32,
            100.0 * (1.0 - e[2] / 100.0).powi(FRAMES as i32 - 1)
        );
        if ratio == 0.5 {
            save_rgb(&picture[0], format!("{dir}/nearest-ratio0.5.png"));
            save_rgb(&picture[1], format!("{dir}/global-ratio0.5.png"));
        }
    }
    println!("(error = true links missed, in % of all links)");
    println!("wrote {dir}/nearest-ratio0.5.png and global-ratio0.5.png (15 frames; green right, red wrong)");
}
