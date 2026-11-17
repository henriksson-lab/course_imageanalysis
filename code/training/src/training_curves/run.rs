// Training curves to diagnose: a small network on a three-armed spiral (three classes),
// healthy and with seven things wrong. Prints the curves and the diagnoses' numbers, and draws
// the curves into out/training-curves/.
//   cargo run -p training -- training-curves                      (your code)
//   cargo run -p training --features solution -- training-curves  (the model solution)

use crate::common::*;
use crate::training_curves::*;

/// One run: the network's layer sizes, the training data, the settings; returns the curve.
fn run(sizes: &[usize], train_set: (&[Vec<f32>], &[usize]), val_set: (&[Vec<f32>], &[usize]), cfg: &Train) -> Vec<Point> {
    let mut net = Mlp::new(sizes, 3);
    train(&mut net, train_set, val_set, cfg, |x, _| x.to_vec()).0
}

fn show(name: &str, curve: &[Point], epochs: &[usize]) {
    let t: Vec<String> = epochs.iter().map(|&e| format!("{:>6.3}", curve[e].train)).collect();
    let v: Vec<String> = epochs.iter().map(|&e| format!("{:>6.3}", curve[e].val)).collect();
    let last = curve.last().unwrap();
    println!("  {name}");
    println!("      training   {}   accuracy {:.0} %", t.join(""), 100.0 * last.train_acc);
    println!("      validation {}   accuracy {:.0} %", v.join(""), 100.0 * last.val_acc);
}

const NOISE: f32 = 0.15;

pub fn main() {
    let dir = "out/training-curves";
    let (xs, ys) = spiral([100, 100, 100], NOISE, 1);
    let (xv, yv) = spiral([100, 100, 100], NOISE, 2);
    let ln3 = chance_loss(&[1, 1, 1]);
    println!("three-armed spiral, 300 training and 300 validation points; guessing gives the loss ln 3 = {ln3:.4}");
    let base = Train { epochs: 200, batch: 32, eta: 0.01, decay: 0.0, class_weight: vec![], seed: 1 };
    let net = [2, 32, 32, 3];
    let epochs = [0, 5, 20, 50, 100, 200];
    println!("loss at epoch{}", epochs.map(|e| format!("{e:>6}")).join(""));

    let mut hnet = Mlp::new(&net, 3);
    let (healthy, batch_losses) = train(&mut hnet, (&xs, &ys), (&xv, &yv), &Train { decay: 0.001, ..base.clone() }, |x, _| x.to_vec());
    show("1. healthy: 2-32-32-3, Adam 0.01, batches of 32, weight decay 0.001", &healthy, &epochs);
    let small = run(&[2, 3], (&xs, &ys), (&xv, &yv), &base);
    show("2. too simple: no hidden layer (a linear classifier)", &small, &epochs);
    let high = run(&net, (&xs, &ys), (&xv, &yv), &Train { eta: 0.3, ..base.clone() });
    show("3. learning rate too high: Adam 0.3", &high, &epochs);
    let low = run(&net, (&xs, &ys), (&xv, &yv), &Train { eta: 0.0001, ..base.clone() });
    show("4. learning rate too low: Adam 0.0001", &low, &epochs);
    let zeros: Vec<Vec<f32>> = xs.iter().map(|x| vec![0.0; x.len()]).collect();
    let zv: Vec<Vec<f32>> = xv.iter().map(|x| vec![0.0; x.len()]).collect();
    let bug = run(&net, (&zeros, &ys), (&zv, &yv), &base);
    show("5. a bug: every input is 0 (the images were never loaded)", &bug, &epochs);
    let mut rng = imgcourse::Rng::new(9);
    let order = shuffled(ys.len(), &mut rng);
    let ys_shuffled: Vec<usize> = order.iter().map(|&i| ys[i]).collect();
    let shuffled_labels = run(&net, (&xs, &ys_shuffled), (&xv, &yv), &base);
    show("6. a bug: the labels are shuffled against the inputs", &shuffled_labels, &epochs);

    // 7. Overfitting: 30 training points, a large network, 1000 epochs; early stopping.
    let (xs30, ys30) = spiral([10, 10, 10], NOISE, 3);
    let over = run(&[2, 64, 64, 3], (&xs30, &ys30), (&xv, &yv), &Train { epochs: 1000, batch: 30, ..base.clone() });
    let e7 = [0, 20, 50, 100, 300, 1000];
    println!("loss at epoch{}", e7.map(|e| format!("{e:>6}")).join(""));
    show("7. overfitting: 30 training points, 2-64-64-3, 1000 epochs", &over, &e7);
    let val: Vec<f32> = over.iter().map(|p| p.val).collect();
    for patience in [10, 50] {
        let (b, stop) = early_stopping(&val, patience);
        println!("      early stopping, patience {patience}: best epoch {b} (validation loss {:.3}, {:.0} % correct), stops at epoch {stop}", val[b], 100.0 * over[b].val_acc);
    }

    // 8. Class imbalance: 240, 50 and 10 points of the three classes.
    let counts = [240, 50, 10];
    let (xi, yi) = spiral(counts, NOISE, 4);
    let (xiv, yiv) = spiral(counts, NOISE, 5);
    println!("8. imbalance: classes 240, 50, 10 (training and validation); always 'class 1' is {:.0} % correct; chance loss {:.3}", 100.0 * 240.0 / 300.0, chance_loss(&counts));
    let w = class_weights(&counts);
    println!("    class weights {:.3}, {:.3}, {:.3}", w[0], w[1], w[2]);
    for (name, cw) in [("plain loss", vec![]), ("weighted loss", w.clone())] {
        let mut m = Mlp::new(&net, 3);
        let (curve, _) = train(&mut m, (&xi, &yi), (&xiv, &yiv), &Train { class_weight: cw, ..base.clone() }, |x, _| x.to_vec());
        let r = m.recalls(&xiv, &yiv, 3);
        let last = curve.last().unwrap();
        println!("    {name}: validation accuracy {:.1} %, loss {:.3}; recall per class {:.0} %, {:.0} %, {:.0} %", 100.0 * last.val_acc, last.val, 100.0 * r[0], 100.0 * r[1], 100.0 * r[2]);
    }
    for (p, g) in [(0.9, 0.0), (0.9, 2.0), (0.5, 0.0), (0.5, 2.0), (0.1, 0.0), (0.1, 2.0)] {
        print!("{}", if g == 0.0 { format!("    focal loss, p = {p}: gamma 0 {:.3}", focal_loss(&[p, 1.0 - p], 0, g)) } else { format!(", gamma 2 {:.4}\n", focal_loss(&[p, 1.0 - p], 0, g)) });
    }

    // 9. The first check: can the network learn 8 examples by heart? With the bug of case 5 it cannot.
    let few: Vec<Vec<f32>> = xs[..8].to_vec();
    let fy = &ys[..8];
    for (name, inputs) in [("8 examples", few.clone()), ("8 examples, inputs 0 (the bug)", vec![vec![0.0, 0.0]; 8])] {
        let mut m = Mlp::new(&net, 3);
        let (c, _) = train(&mut m, (&inputs, fy), (&inputs, fy), &Train { epochs: 300, batch: 8, ..base.clone() }, |x, _| x.to_vec());
        println!("9. overfit one batch, {name}: loss {:.3} -> {:.4} after 300 steps ({:.0} % correct)", c[0].train, c[300].train, 100.0 * c[300].train_acc);
    }

    // Pictures: training (blue) and validation (red) loss, 0 to 1.6, and the smoothed curve of case 3.
    let pairs: [(&str, &[Point]); 7] = [
        ("1-healthy", &healthy),
        ("2-too-simple", &small),
        ("3-lr-high", &high),
        ("4-lr-low", &low),
        ("5-zero-inputs", &bug),
        ("6-shuffled-labels", &shuffled_labels),
        ("7-overfitting", &over),
    ];
    for (name, c) in pairs {
        let t: Vec<f32> = c.iter().map(|p| p.train).collect();
        let v: Vec<f32> = c.iter().map(|p| p.val).collect();
        plot_curves(&[(&t, BLUE), (&v, RED)], (0.0, 1.6), 0.4, &format!("{dir}/{name}.png"));
    }
    let sm = smooth(&batch_losses, 0.95);
    let at = [0, 50, 100, 500, 1000, 1999];
    println!("case 1, the loss of each mini-batch ({} steps) and smoothed with beta 0.95:", batch_losses.len());
    println!("      step     {}", at.map(|t| format!("{t:>7}")).join(""));
    println!("      raw      {}", at.map(|t| format!("{:>7.3}", batch_losses[t])).join(""));
    println!("      smoothed {}", at.map(|t| format!("{:>7.3}", sm[t])).join(""));
    let spread = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min)..v.iter().cloned().fold(0.0, f32::max);
    let (r, q) = (spread(&batch_losses[1000..]), spread(&sm[1000..]));
    println!("      steps 1000-1999: raw between {:.3} and {:.3}, smoothed between {:.3} and {:.3}", r.start, r.end, q.start, q.end);
    plot_curves(&[(&batch_losses, BLUE), (&sm, ORANGE)], (0.0, 1.6), 0.4, &format!("{dir}/1-healthy-batches.png"));
    let csv: String = std::iter::once("step,batch_loss,smoothed\n".to_string())
        .chain(batch_losses.iter().zip(&sm).enumerate().map(|(t, (b, q))| format!("{t},{b:.4},{q:.4}\n")))
        .collect();
    std::fs::write(format!("{dir}/batch-losses.csv"), csv).expect("write the table");
    println!("wrote the plots to {dir}/ (blue training, red validation, loss 0 to 1.6, grid every 0.4)");
}
