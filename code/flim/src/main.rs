// The programs of this day, one per unit:  cargo run -p flim -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Fluorescence lifetime imaging (FLIM). Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "flim")]
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
        Unit::Lifetime => flim::lifetime::run::main(),
        Unit::Decayfit => flim::decayfit::run::main(),
        Unit::Phasor => flim::phasor::run::main(),
        Unit::FlimFret => flim::flim_fret::run::main(),
    }
}
