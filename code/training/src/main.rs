// The programs of this day, one per unit:  cargo run -p training -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use training::*;

/// Training networks in practice. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "training")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Optimisers and learning rates
    #[command(name = "optimisers")]
    Optimisers,
    /// Initialisation, normalisation and residual connections
    #[command(name = "normalisation")]
    Normalisation,
    /// Regularisation and data augmentation
    #[command(name = "augmentation")]
    Augmentation,
    /// Reading training curves
    #[command(name = "training-curves")]
    TrainingCurves,
}

fn main() {
    match Cli::parse().unit {
        Unit::Optimisers => optimisers::run::main(),
        Unit::Normalisation => normalisation::run::main(),
        Unit::Augmentation => augmentation::run::main(),
        Unit::TrainingCurves => training_curves::run::main(),
    }
}
