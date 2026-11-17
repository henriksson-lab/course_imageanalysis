// The programs of this day, one per unit:  cargo run -p radon -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Radon transform and tomography. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "radon")]
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
        Unit::Radon => radon::radon::run::main(),
        Unit::Backprojection => radon::backprojection::run::main(),
        Unit::Fbp => radon::fbp::run::main(),
        Unit::AlgebraicRecon => radon::algebraic_recon::run::main(),
    }
}
