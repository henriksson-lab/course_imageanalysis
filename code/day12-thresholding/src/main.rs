// The programs of day 12, one per unit:  cargo run -p day12 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day12::*;

/// Day 12: Thresholding. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day12")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Histograms & a global threshold
    #[command(name = "histogram")]
    Histogram,
    /// Choosing the threshold: Otsu & minimum error
    #[command(name = "otsu")]
    Otsu,
    /// Local thresholds & hysteresis
    #[command(name = "local-threshold")]
    LocalThreshold,
    /// Evaluation: Dice, ROC & precision–recall
    #[command(name = "seg-eval")]
    SegEval,
}

fn main() {
    match Cli::parse().unit {
        Unit::Histogram => histogram::run::main(),
        Unit::Otsu => otsu::run::main(),
        Unit::LocalThreshold => local_threshold::run::main(),
        Unit::SegEval => seg_eval::run::main(),
    }
}
