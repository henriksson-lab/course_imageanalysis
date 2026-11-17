// The programs of day 18, one per unit:  cargo run -p day18 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day18::*;

/// Day 18: Moments. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day18")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Moments as projections onto monomials
    #[command(name = "moments-basis")]
    MomentsBasis,
    /// Normalisation and moment invariants
    #[command(name = "moment-invariants")]
    MomentInvariants,
    /// Better bases: Legendre moments and reconstruction
    #[command(name = "orthogonal-moments")]
    OrthogonalMoments,
    /// Zernike moments on the unit disc
    #[command(name = "zernike")]
    Zernike,
}

fn main() {
    match Cli::parse().unit {
        Unit::MomentsBasis => moments_basis::run::main(),
        Unit::MomentInvariants => moment_invariants::run::main(),
        Unit::OrthogonalMoments => orthogonal_moments::run::main(),
        Unit::Zernike => zernike::run::main(),
    }
}
