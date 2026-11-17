// The programs of day 7, one per unit:  cargo run -p day07 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day07::*;

/// Day 7: Wavelets. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day07")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Localised basis functions and the Haar transform
    #[command(name = "haar")]
    Haar,
    /// Multiresolution, filter banks and Daubechies wavelets
    #[command(name = "filterbank")]
    Filterbank,
    /// The 2D wavelet transform and sparsity of images
    #[command(name = "dwt2d")]
    Dwt2d,
    /// Denoising by wavelet shrinkage
    #[command(name = "shrinkage")]
    Shrinkage,
}

fn main() {
    match Cli::parse().unit {
        Unit::Haar => haar::run::main(),
        Unit::Filterbank => filterbank::run::main(),
        Unit::Dwt2d => dwt2d::run::main(),
        Unit::Shrinkage => shrinkage::run::main(),
    }
}
