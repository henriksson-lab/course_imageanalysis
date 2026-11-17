// Active learning on a pool of 500 cells in a 2D feature space: which cell to label next?
// Learning curves (test accuracy against the number of labels, mean of 5 starts) for random
// choice, the most uncertain cell, the most different cell, and both in turn; and a picture of
// the pool after 40 labels with each rule.
//   cargo run -p finetuning -- few-labels                      (your code)
//   cargo run -p finetuning --features solution -- few-labels  (the model solution)

use crate::few_labels::*;
use imgcourse::*;

#[derive(Clone, Copy, PartialEq)]
enum Rule {
    Random,
    Uncertain,
    Different,
    Both,
}

/// Label cells one at a time with a rule until `n` are labelled; the model after each label.
fn run_rule(pool: &[Cell], start: &[usize], rule: Rule, n: usize, seed: u64) -> (Vec<usize>, Vec<Model>) {
    let mut rng = Rng::new(seed);
    let mut idx = start.to_vec();
    let mut labelled = vec![false; pool.len()];
    for &i in start {
        labelled[i] = true;
    }
    let mut models = vec![Model::fit(pool, &idx)];
    while idx.len() < n {
        let m = models.last().unwrap();
        let probs: Vec<f32> = pool.iter().map(|c| m.prob(c.x, c.y)).collect();
        let turn = idx.len() - start.len();
        let next = match rule {
            Rule::Random => loop {
                let j = rng.below(pool.len());
                if !labelled[j] {
                    break Some(j);
                }
            },
            Rule::Uncertain => most_uncertain(&probs, &labelled),
            Rule::Different => most_different(pool, &labelled),
            Rule::Both => {
                if turn.is_multiple_of(2) {
                    most_uncertain(&probs, &labelled)
                } else {
                    most_different(pool, &labelled)
                }
            }
        };
        let Some(j) = next else { break };
        labelled[j] = true;
        idx.push(j);
        models.push(Model::fit(pool, &idx));
    }
    (idx, models)
}

fn picture(pool: &[Cell], idx: &[usize], m: &Model) -> Image<[f32; 3]> {
    let s = 300;
    let mut img = Image::from_fn(s, s, |x, y| {
        let p = m.prob(x as f32 / s as f32, 1.0 - y as f32 / s as f32);
        let t = (p - 0.5).abs() * 0.9;
        if p > 0.5 {
            [1.0, 1.0 - 0.35 * t, 1.0 - 0.8 * t]
        } else {
            [1.0 - 0.7 * t, 1.0 - 0.4 * t, 1.0]
        }
    });
    let mut dot = |c: &Cell, r: isize, col: [f32; 3]| {
        let (cx, cy) = ((c.x * s as f32) as isize, ((1.0 - c.y) * s as f32) as isize);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r && img.contains(cx + dx, cy + dy) {
                    img[((cx + dx) as usize, (cy + dy) as usize)] = col;
                }
            }
        }
    };
    let colour = |c: &Cell| if c.class == 1 { [0.88, 0.54, 0.12] } else { [0.16, 0.44, 0.59] };
    for c in pool {
        dot(c, 1, colour(c));
    }
    for &i in idx {
        dot(&pool[i], 5, [0.0, 0.0, 0.0]);
        dot(&pool[i], 3, colour(&pool[i]));
    }
    img
}

pub fn main() {
    let dir = "out/few-labels";
    let pool = make_pool(500, 1);
    let test = make_pool(600, 2);
    println!("uncertainty of p = 0.5, 0.7, 0.9, 0.99: {:.2} {:.2} {:.2} {:.2}", uncertainty(0.5), uncertainty(0.7), uncertainty(0.9), uncertainty(0.99));
    println!("entropy (bits) of the same:           {:.2} {:.2} {:.2} {:.2}", entropy(0.5), entropy(0.7), entropy(0.9), entropy(0.99));
    let all: Vec<usize> = (0..pool.len()).collect();
    let (acc, rare) = Model::fit(&pool, &all).score(&test);
    println!("all 500 cells labelled: test accuracy {:.1} %, rare subtype {:.0} %", 100.0 * acc, 100.0 * rare);
    let ns = [4, 6, 10, 15, 20, 30, 40];
    println!("\ntest accuracy (rare subtype correct), mean of 5 different starts");
    println!("  labels:            {}", ns.iter().map(|n| format!("{n:>11}")).collect::<String>());
    for (name, rule) in [("random", Rule::Random), ("most uncertain", Rule::Uncertain), ("most different", Rule::Different), ("both in turn", Rule::Both)] {
        let mut acc = [0.0f32; 7];
        let mut rare = [0.0f32; 7];
        for s in 1..=5 {
            let (_, models) = run_rule(&pool, &start_set(&pool, s), rule, 40, 100 + s);
            for (k, &n) in ns.iter().enumerate() {
                let (a, r) = models[n - 4].score(&test);
                acc[k] += a / 5.0;
                rare[k] += r / 5.0;
            }
        }
        let cells: String = (0..7).map(|k| format!("  {:4.1} ({:3.0})", 100.0 * acc[k], 100.0 * rare[k])).collect();
        println!("  {name:17} {cells}");
        let (idx, models) = run_rule(&pool, &start_set(&pool, 1), rule, 40, 101);
        let file = name.replace(' ', "-");
        save_rgb(&picture(&pool, &idx, models.last().unwrap()), format!("{dir}/{file}-40.png"));
    }
    println!("wrote {dir}/<rule>-40.png: the pool after 40 labels (big dots), background = the model's prediction");
}
