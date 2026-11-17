// The programs of this day, one per unit:  cargo run -p moments -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Moments. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "moments")]
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
        Unit::MomentsBasis => moments::moments_basis::run::main(),
        Unit::MomentInvariants => moments::moment_invariants::run::main(),
        Unit::OrthogonalMoments => moments::orthogonal_moments::run::main(),
        Unit::Zernike => moments::zernike::run::main(),
    }
}
