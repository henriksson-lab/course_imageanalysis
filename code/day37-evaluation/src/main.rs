// The programs of day 37, one per unit:  cargo run -p day37 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day37::*;

/// Day 37: Evaluation and statistics. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day37")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Matching objects: IoU, F1 and average precision
    #[command(name = "object-eval")]
    ObjectEval,
    /// Boundaries, counts and the uncertain ground truth
    #[command(name = "agreement")]
    Agreement,
    /// Honest validation: leakage, splits and domain shift
    #[command(name = "leakage")]
    Leakage,
    /// From measurements to claims
    #[command(name = "replicates")]
    Replicates,
}

fn main() {
    match Cli::parse().unit {
        Unit::ObjectEval => object_eval::run::main(),
        Unit::Agreement => agreement::run::main(),
        Unit::Leakage => leakage::run::main(),
        Unit::Replicates => replicates::run::main(),
    }
}
