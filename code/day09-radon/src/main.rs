// The programs of day 9, one per unit:  cargo run -p day09 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day09::*;

/// Day 9: Radon transform and tomography. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day09")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Projections and the sinogram
    #[command(name = "radon")]
    Radon,
    /// Back-projection and the Fourier slice theorem
    #[command(name = "backprojection")]
    Backprojection,
    /// Filtered back-projection and sampling
    #[command(name = "fbp")]
    Fbp,
    /// Reconstruction as a linear system
    #[command(name = "algebraic-recon")]
    AlgebraicRecon,
}

fn main() {
    match Cli::parse().unit {
        Unit::Radon => radon::run::main(),
        Unit::Backprojection => backprojection::run::main(),
        Unit::Fbp => fbp::run::main(),
        Unit::AlgebraicRecon => algebraic_recon::run::main(),
    }
}
