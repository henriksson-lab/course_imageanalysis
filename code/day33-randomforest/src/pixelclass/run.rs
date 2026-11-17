// Classifies every pixel of a synthetic fluorescence image from a few "painted" strokes,
// with a forest on your feature stack; results in out/pixelclass/.
//   cargo run -p day33 -- pixelclass                      (your code)
//   cargo run -p day33 --features solution -- pixelclass  (the model solution)

use imgcourse::*;
use crate::pixelclass::*;

/// The test image: a dark background whose brightness rises to the right, smooth nuclei
/// (class 1) and granular cells (class 2) that are about as bright as the nuclei but
/// speckled. Returns the image, the true class of every pixel, and an object number for every
/// pixel (0 = background).
fn test_image() -> (Image<f32>, Image<usize>, Image<u32>) {
    let (nuclei, nucleus_labels) = samples::nuclei(256, 10, 21);
    let (_, granular_labels) = samples::nuclei(256, 8, 32);
    let mut rng = Rng::new(5);
    let speckle = Image::from_fn(256, 256, |_, _| rng.uniform());
    let class = Image::from_fn(256, 256, |x, y| {
        if nucleus_labels[(x, y)] > 0 {
            1
        } else if granular_labels[(x, y)] > 0 {
            2
        } else {
            0
        }
    });
    let objects = Image::from_fn(256, 256, |x, y| match class[(x, y)] {
        1 => nucleus_labels[(x, y)],
        2 => 100 + granular_labels[(x, y)],
        _ => 0,
    });
    let img = Image::from_fn(256, 256, |x, y| {
        let granular = if class[(x, y)] == 2 { 0.65 * (0.3 + 1.4 * speckle[(x, y)]) } else { 0.0 };
        nuclei[(x, y)] + granular + 0.3 * x as f32 / 256.0
    });
    (img, class, objects)
}

/// Strokes, as a painter would draw them on the left half of the image: a short horizontal
/// stroke through the middle of the first three objects of each class, and some longer strokes
/// through the background, away from every object. Returns `(x, y, class)` for every pixel.
fn paint(class: &Image<usize>, objects: &Image<u32>) -> Vec<(usize, usize, usize)> {
    let (w, h) = (class.width(), class.height());
    let mut strokes = Vec::new();
    let mut done: Vec<u32> = Vec::new();
    for (x, y) in objects.coords() {
        let (o, c) = (objects[(x, y)], class[(x, y)]);
        let painted_of_class = done.iter().filter(|&&d| (d >= 100) == (c == 2)).count();
        if o == 0 || x >= w / 2 || done.contains(&o) || painted_of_class == 3 {
            continue;
        }
        // the middle row of this object, and its extent along that row
        let pixels: Vec<(usize, usize)> = objects.coords().filter(|&p| objects[p] == o).collect();
        let cy = pixels.iter().map(|p| p.1).sum::<usize>() / pixels.len();
        let row: Vec<usize> = pixels.iter().filter(|p| p.1 == cy).map(|p| p.0).collect();
        let (x0, x1) = (*row.iter().min().unwrap(), *row.iter().max().unwrap());
        for yy in cy - 1..=cy + 1 {
            for xx in x0 + 3..x1.saturating_sub(2) {
                if objects[(xx, yy)] == o {
                    strokes.push((xx, yy, c));
                }
            }
        }
        done.push(o);
    }
    let near_object = |x: usize, y: usize| {
        (-6..=6isize).any(|dy| (-6..=6isize).any(|dx| objects.get_border(x as isize + dx, y as isize + dy, Border::Clamp) > 0))
    };
    for &y0 in &[h / 8, h / 3, 2 * h / 3, 7 * h / 8] {
        for y in y0..y0 + 3 {
            for x in 8..w / 2 - 8 {
                if !near_object(x, y) {
                    strokes.push((x, y, 0));
                }
            }
        }
    }
    strokes
}

struct Scores {
    oob_accuracy: f32,
    /// accuracy, Dice of the nuclei, Dice of the granular cells
    left: (f32, f32, f32),
    right: (f32, f32, f32),
}

/// Train a forest on the painted pixels with the given channels, classify every pixel, and
/// score the result against the truth on the left (painted) and right (held-out) halves.
fn run(stack: &[Image<f32>], strokes: &[(usize, usize, usize)], truth: &Image<usize>, balance: bool, name: &str) -> Scores {
    let points: Vec<Vec<f32>> = strokes.iter().map(|&(x, y, _)| stack.iter().map(|c| c[(x, y)]).collect()).collect();
    let labels: Vec<usize> = strokes.iter().map(|s| s.2).collect();
    let counts: Vec<usize> = (0..3).map(|k| labels.iter().filter(|&&l| l == k).count()).collect();
    let class_weights = if balance { crate::pixelclass::class_weights(&counts) } else { vec![1.0; 3] };
    let m = (stack.len() as f32).sqrt().round() as usize;
    let forest = Forest::train(&points, &labels, 3, &Settings { n_trees: 50, m, class_weights, seed: 1 });

    let (w, h) = (truth.width(), truth.height());
    let p = Image::from_fn(w, h, |x, y| {
        let q = forest.predict(&stack.iter().map(|c| c[(x, y)]).collect::<Vec<f32>>());
        [q[0], q[1], q[2]]
    });
    let predicted = p.map(|p| argmax(&p));
    let dir = "out/pixelclass";
    save_gray(&p.map(|p| p[1]), format!("{dir}/{name}-p-nucleus.png"));
    save_gray(&p.map(|p| 1.0 - p.iter().fold(0.0f32, |a, &b| a.max(b))), format!("{dir}/{name}-uncertainty.png"));
    save_labels(&predicted.map(|c| [0, 2, 3][c]), format!("{dir}/{name}-segmentation.png"));

    // scores on one half of the image: accuracy and Dice of the nucleus class
    let half = |left: bool| {
        let (mut t, mut p) = (Vec::new(), Vec::new());
        for (x, y) in truth.coords() {
            if (x < w / 2) == left {
                t.push(truth[(x, y)]);
                p.push(predicted[(x, y)]);
            }
        }
        let m = confusion(&t, &p, 3);
        ((m[0][0] + m[1][1] + m[2][2]) as f32 / t.len() as f32, dice(&m, 1), dice(&m, 2))
    };
    Scores { oob_accuracy: 1.0 - forest.oob_error(&points, &labels), left: half(true), right: half(false) }
}

pub fn main() {
    let dir = "out/pixelclass";
    let start = std::time::Instant::now();
    let (img, truth, objects) = test_image();
    let strokes = paint(&truth, &objects);
    let n: Vec<usize> = (0..3).map(|k| strokes.iter().filter(|s| s.2 == k).count()).collect();
    println!("painted (all on the left half): {} background, {} nucleus, {} granular-cell pixels", n[0], n[1], n[2]);

    save_gray(&img, format!("{dir}/input.png"));
    save_labels(&truth.map(|c| [0, 2, 3][c]), format!("{dir}/truth.png"));
    let mut painted = gray_to_rgb(&img);
    for &(x, y, c) in &strokes {
        painted[(x, y)] = [[0.2, 0.5, 1.0], [1.0, 0.5, 0.0], [0.2, 0.9, 0.2]][c];
    }
    save_rgb(&painted, format!("{dir}/strokes.png"));

    let sigmas = [0.7, 1.6, 3.5, 8.0];
    let stack = feature_stack(&img, &sigmas);
    let intensity_only: Vec<Image<f32>> = stack.iter().step_by(3).cloned().collect();
    let only_small: Vec<Image<f32>> = stack[..6].to_vec();
    println!("{:<35} {:>6}   {:<27}  {:<27}", "", "OOB", "left half (painted)", "right half (held out)");
    println!("{:<35} {:>6}   {:<27}  {:<27}", "", "", "acc     Dice N  Dice G", "acc     Dice N  Dice G");
    for (name, file, channels, balance) in [
        ("full stack", "full-stack", &stack, false),
        ("full stack, balanced", "balanced", &stack, true),
        ("intensity only", "intensity-only", &intensity_only, false),
        ("small scales only (σ ≤ 1.6)", "small-scales", &only_small, false),
    ] {
        let s = run(channels, &strokes, &truth, balance, file);
        println!(
            "{name:<30} ({:2}) {:6.1} %   {:5.1} %  {:.2}    {:.2}    {:5.1} %  {:.2}    {:.2}",
            channels.len(),
            100.0 * s.oob_accuracy,
            100.0 * s.left.0,
            s.left.1,
            s.left.2,
            100.0 * s.right.0,
            s.right.1,
            s.right.2
        );
    }
    println!("(took {:.1} s)", start.elapsed().as_secs_f64());
    println!("wrote the results to {dir}/");
}
