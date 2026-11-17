// Augmentations of the course micrograph, then a small network trained on 20 crops of cells
// (round or elongated) with and without augmentation and weight decay, printing its training
// and validation curves. Pictures in out/augmentation/.
//   cargo run -p training -- augmentation                      (your code)
//   cargo run -p training --features solution -- augmentation  (the model solution)

use crate::augmentation::*;
use crate::common::*;
use imgcourse::*;

/// Several 16 × 16 crops side by side (2 px apart), enlarged 4 times, for looking at.
fn strip(imgs: &[Image<f32>]) -> Image<f32> {
    let n = imgs.len();
    Image::from_fn(n * 66 - 2, 64, |x, y| {
        let (k, xx) = (x / 66, x % 66);
        if xx >= 64 {
            1.0
        } else {
            imgs[k][(xx / 4, y / 4)]
        }
    })
}

pub fn main() {
    let dir = "out/augmentation";
    // 1. The augmentations on the micrograph.
    let cell = samples::cell();
    save_gray(&flip_horizontal(&cell), format!("{dir}/cell-flipped.png"));
    save_gray(&rotate(&cell, 30.0), format!("{dir}/cell-rotated30.png"));
    save_gray(&elastic(&cell, 4.0, 10.0, 1), format!("{dir}/cell-elastic.png"));
    save_gray(&intensity(&cell, 0.8, 0.1, 0.6), format!("{dir}/cell-intensity.png"));
    let mut rng = Rng::new(3);
    let a = Augment { flips: true, max_angle: 180.0, gain_range: 0.3, noise: 0.05 };
    let copies: Vec<Image<f32>> = (0..6).map(|_| augment(&cell.crop(64, 64, 128, 128), &a, &mut rng)).collect();
    for (k, c) in copies.iter().enumerate() {
        save_gray(c, format!("{dir}/random-{k}.png"));
    }
    let moved = rotate(&cell, 30.0);
    println!("micrograph: mean grey value {:.4}; rotated by 30 degrees {:.4}; flipped {:.4}", cell.mean(), moved.mean(), flip_horizontal(&cell).mean());

    // 2. The crops: 20 for training (elongated cells all lying), 200 for validation (any direction, any brightness).
    let (ti, ty) = crops(Task::Shape, 20, true, 11);
    let (vi, vy) = crops(Task::Shape, 200, false, 12);
    save_gray(&strip(&ti[..8]), format!("{dir}/train-crops.png"));
    save_gray(&strip(&vi[..8]), format!("{dir}/val-crops.png"));
    let tx: Vec<Vec<f32>> = ti.iter().map(to_input).collect();
    let vx: Vec<Vec<f32>> = vi.iter().map(to_input).collect();

    let settings: [(&str, Option<Augment>, f32); 5] = [
        ("no augmentation", None, 0.0),
        ("weight decay 0.01", None, 0.01),
        ("flips + rotations", Some(Augment { flips: true, max_angle: 180.0, gain_range: 0.0, noise: 0.0 }), 0.0),
        ("flips + rotations + gain 0.4 + noise 0.03", Some(Augment { flips: true, max_angle: 180.0, gain_range: 0.4, noise: 0.03 }), 0.0),
        ("gain 0.4 + noise 0.03 only", Some(Augment { flips: false, max_angle: 0.0, gain_range: 0.4, noise: 0.03 }), 0.0),
    ];
    println!("cells, round vs elongated: 20 training crops (elongated ones lying), 200 validation crops (any direction and brightness)");
    println!("network 256 -> 32 -> 2, Adam 0.003, batches of 10, 300 epochs; loss (accuracy) at epoch 0, 25, 100, 300; lowest validation loss");
    let mut plots = Vec::new();
    for (name, aug, decay) in settings {
        let mut net = Mlp::new(&[256, 32, 2], 5);
        let cfg = Train { epochs: 300, batch: 10, eta: 0.003, decay, class_weight: vec![], seed: 7 };
        let (curve, _) = train(&mut net, (&tx, &ty), (&vx, &vy), &cfg, |x, rng| match aug {
            Some(a) => to_input(&augment(&from_input(x), &a, rng)),
            None => x.to_vec(),
        });
        println!("  {name}:");
        let at = |e: usize| format!("{:.3} ({:.0} %)", curve[e].train, 100.0 * curve[e].train_acc);
        let atv = |e: usize| format!("{:.3} ({:.0} %)", curve[e].val, 100.0 * curve[e].val_acc);
        println!("      training   {}", [0, 25, 100, 300].map(at).join("  "));
        println!("      validation {}", [0, 25, 100, 300].map(atv).join("  "));
        let b = best(&curve);
        println!("      lowest validation loss {:.3} at epoch {} ({:.0} % correct)", b.val, b.epoch, 100.0 * b.val_acc);
        plots.push(curve);
    }
    let tr = |k: usize| plots[k].iter().map(|p| p.train).collect::<Vec<f32>>();
    let va = |k: usize| plots[k].iter().map(|p| p.val).collect::<Vec<f32>>();
    let (t0, v0, t2, v2) = (tr(0), va(0), tr(3), va(3));
    plot_curves(&[(&t0, BLUE), (&v0, RED), (&t2, GREEN), (&v2, ORANGE)], (0.0, 2.0), 0.5, &format!("{dir}/curves.png"));

    // 3. When augmentation is wrong: the class is the direction itself.
    let (ti, ty) = crops(Task::Orientation, 20, true, 21);
    let (vi, vy) = crops(Task::Orientation, 200, false, 22);
    let tx: Vec<Vec<f32>> = ti.iter().map(to_input).collect();
    let vx: Vec<Vec<f32>> = vi.iter().map(to_input).collect();
    println!("cells lying vs standing (the class is the direction): validation accuracy after 300 epochs");
    for (name, aug) in [
        ("no augmentation", None),
        ("flips only", Some(Augment { flips: true, max_angle: 0.0, gain_range: 0.0, noise: 0.0 })),
        ("rotations up to 180 degrees", Some(Augment { flips: false, max_angle: 180.0, gain_range: 0.0, noise: 0.0 })),
    ] {
        let mut net = Mlp::new(&[256, 32, 2], 5);
        let cfg = Train { epochs: 300, batch: 10, eta: 0.003, decay: 0.0, class_weight: vec![], seed: 7 };
        let (curve, _) = train(&mut net, (&tx, &ty), (&vx, &vy), &cfg, |x, rng| match aug {
            Some(a) => to_input(&augment(&from_input(x), &a, rng)),
            None => x.to_vec(),
        });
        let last = curve.last().unwrap();
        println!("    {name}: {:.0} % (training {:.0} %)", 100.0 * last.val_acc, 100.0 * last.train_acc);
    }
    println!("wrote the pictures to {dir}/ (curves.png: blue/red training/validation without augmentation, green/orange with flips + rotations + gain + noise; loss 0 to 2)");
}
