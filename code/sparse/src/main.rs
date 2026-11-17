// The programs of this day, one per unit:  cargo run -p sparse -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Sparse representations. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "sparse")]
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
        Unit::Sparsity => sparse::sparsity::run::main(),
        Unit::Ista => sparse::ista::run::main(),
        Unit::CompressedSensing => sparse::compressed_sensing::run::main(),
        Unit::CsMri => sparse::cs_mri::run::main(),
        Unit::Dictionaries => sparse::dictionaries::run::main(),
    }
}
