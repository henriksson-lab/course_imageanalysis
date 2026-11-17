// The programs of this day, one per unit:  cargo run -p evaluation -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Evaluation and statistics. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "evaluation")]
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
        Unit::ObjectEval => evaluation::object_eval::run::main(),
        Unit::Agreement => evaluation::agreement::run::main(),
        Unit::Leakage => evaluation::leakage::run::main(),
        Unit::Replicates => evaluation::replicates::run::main(),
    }
}
