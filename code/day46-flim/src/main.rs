// The programs of day 46, one per unit:  cargo run -p day46 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day46::*;

/// Day 46: Fluorescence lifetime imaging (FLIM). Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day46")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Lifetimes, decays and photon counting
    #[command(name = "lifetime")]
    Lifetime,
    /// Fitting decays with few photons
    #[command(name = "decayfit")]
    Decayfit,
    /// Frequency domain and phasors
    #[command(name = "phasor")]
    Phasor,
    /// FLIM-FRET and metabolic imaging
    #[command(name = "flim-fret")]
    FlimFret,
}

fn main() {
    match Cli::parse().unit {
        Unit::Lifetime => lifetime::run::main(),
        Unit::Decayfit => decayfit::run::main(),
        Unit::Phasor => phasor::run::main(),
        Unit::FlimFret => flim_fret::run::main(),
    }
}
