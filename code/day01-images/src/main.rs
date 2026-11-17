// The programs of day 1, one per unit:  cargo run -p day01 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day01::*;

/// Day 1: Images, filters and frequencies. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day01")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Images as arrays & geometric transforms
    #[command(name = "images")]
    Images,
    /// Filtering & convolution
    #[command(name = "kernels")]
    Kernels,
    /// Images in frequency space
    #[command(name = "fourier")]
    Fourier,
    /// Principal component analysis
    #[command(name = "pca")]
    Pca,
}

fn main() {
    match Cli::parse().unit {
        Unit::Images => images::run::main(),
        Unit::Kernels => kernels::run::main(),
        Unit::Fourier => fourier::run::main(),
        Unit::Pca => pca::run::main(),
    }
}
