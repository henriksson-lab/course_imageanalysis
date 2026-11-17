// The programs of day 11, one per unit:  cargo run -p day11 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day11::*;

/// Day 11: Sparse representations. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day11")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Sparsity and the ℓ1 penalty
    #[command(name = "sparsity")]
    Sparsity,
    /// Solving with ℓ1: thresholding iterations and pursuit
    #[command(name = "ista")]
    Ista,
    /// Compressed sensing
    #[command(name = "compressed-sensing")]
    CompressedSensing,
    /// Fast MRI with compressed sensing
    #[command(name = "cs-mri")]
    CsMri,
    /// Sparse coding and dictionary learning
    #[command(name = "dictionaries")]
    Dictionaries,
}

fn main() {
    match Cli::parse().unit {
        Unit::Sparsity => sparsity::run::main(),
        Unit::Ista => ista::run::main(),
        Unit::CompressedSensing => compressed_sensing::run::main(),
        Unit::CsMri => cs_mri::run::main(),
        Unit::Dictionaries => dictionaries::run::main(),
    }
}
