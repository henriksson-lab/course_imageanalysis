// The programs of this day, one per unit:  cargo run -p images -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Images, filters and frequencies. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "images")]
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
        Unit::Images => images::images::run::main(),
        Unit::Kernels => images::kernels::run::main(),
        Unit::Fourier => images::fourier::run::main(),
        Unit::Pca => images::pca::run::main(),
    }
}
