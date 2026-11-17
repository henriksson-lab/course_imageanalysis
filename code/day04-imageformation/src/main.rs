// The programs of day 4, one per unit:  cargo run -p day04 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day04::*;

/// Day 4: Image formation. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day04")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Image formation and the point spread function
    #[command(name = "psf")]
    Psf,
    /// Resolution, sampling and measuring the PSF
    #[command(name = "resolution")]
    Resolution,
    /// Running the model backwards: an ill-posed problem
    #[command(name = "illposed")]
    Illposed,
}

fn main() {
    match Cli::parse().unit {
        Unit::Psf => psf::run::main(),
        Unit::Resolution => resolution::run::main(),
        Unit::Illposed => illposed::run::main(),
    }
}
