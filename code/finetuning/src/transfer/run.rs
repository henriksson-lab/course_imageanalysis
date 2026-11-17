// Transfer learning on the toy patches: a network pretrained on the old task (four shapes)
// is reused for the new task (three cell phenotypes) with few labels. Pictures in out/transfer/.
//   cargo run -p finetuning -- transfer                      (your code)
//   cargo run -p finetuning --features solution -- transfer  (the model solution)

use crate::transfer::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/transfer";
    println!("pretraining on the old task (2400 labelled patches of four shapes)...");
    let pre = pretrained();
    let (old_x, old_y) = make_set(true, 400, 202, LOOK_A);
    println!("old task: accuracy {:.1} % on 400 new patches", 100.0 * net_accuracy(&pre, &old_x, &old_y));
    let random = new_net(&[P * P, 32, 16, 4], 99);
    let three = with_new_head(&pre, 3, 11);
    println!(
        "weights trained: frozen + new head {}, last layers {}, all layers {}",
        trained_weights(&three, 2),
        trained_weights(&three, 1),
        trained_weights(&three, 0)
    );

    for (name, look) in [("same microscope", LOOK_A), ("other modality", LOOK_FAR)] {
        let (pool_x, pool_y) = make_set(false, 300, 303, look);
        let (val_x, val_y) = make_set(false, 300, 404, look);
        let val_pre: Vec<Vec<f32>> = val_x.iter().map(|x| features(&pre, x, 2)).collect();
        let val_rnd: Vec<Vec<f32>> = val_x.iter().map(|x| features(&random, x, 2)).collect();
        println!("\nnew task, {name}: validation accuracy (300 patches)");
        println!("  labels   probe on pretrained   probe on random net   all layers tuned   from scratch");
        for n in [6, 15, 30, 90, 300] {
            let (xs, ys) = (&pool_x[..n], &pool_y[..n]);
            let f_pre: Vec<Vec<f32>> = xs.iter().map(|x| features(&pre, x, 2)).collect();
            let f_rnd: Vec<Vec<f32>> = xs.iter().map(|x| features(&random, x, 2)).collect();
            let probe = train_probe(&f_pre, ys, 3, 500, 0.1);
            let probe_rnd = train_probe(&f_rnd, ys, 3, 500, 0.1);
            let mut tuned = with_new_head(&pre, 3, 11);
            train(&mut tuned, xs, ys, &[0.02, 0.02, 0.02], 500, 0, 5);
            let mut scratch = new_net(&[P * P, 32, 16, 3], 111);
            train(&mut scratch, xs, ys, &[0.02, 0.02, 0.02], 500, 0, 5);
            println!(
                "  {n:6}   {:17.1} %   {:17.1} %   {:14.1} %   {:10.1} %",
                100.0 * probe_accuracy(&probe, &val_pre, &val_y),
                100.0 * probe_accuracy(&probe_rnd, &val_rnd, &val_y),
                100.0 * net_accuracy(&tuned, &val_x, &val_y),
                100.0 * net_accuracy(&scratch, &val_x, &val_y)
            );
        }
    }

    // pictures: the 32 first-layer filters, and example patches of both tasks
    let filters: Vec<Image<f32>> = pre[0].w.iter().map(|row| filter_image(row)).collect();
    save_gray(&mosaic(&filters, 8, 8), format!("{dir}/filters.png"));
    let mut tiles = Vec::new();
    for (old, look) in [(true, LOOK_A), (false, LOOK_A), (false, LOOK_FAR)] {
        let (xs, _) = make_set(old, 8, 77, look);
        tiles.extend(xs.iter().map(|x| Image::from_fn(P, P, |i, j| x[j * P + i])));
    }
    save_gray(&mosaic(&tiles, 8, 6), format!("{dir}/patches.png"));
    println!("\nwrote {dir}/filters.png (first-layer weights, grey = 0) and {dir}/patches.png (rows: old task, new task, other modality)");
}
