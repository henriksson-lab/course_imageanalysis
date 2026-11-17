// The programs of this day, one per unit:  cargo run -p thresholding -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Thresholding. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "thresholding")]
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
        Unit::Histogram => thresholding::histogram::run::main(),
        Unit::Otsu => thresholding::otsu::run::main(),
        Unit::LocalThreshold => thresholding::local_threshold::run::main(),
        Unit::SegEval => thresholding::seg_eval::run::main(),
    }
}
