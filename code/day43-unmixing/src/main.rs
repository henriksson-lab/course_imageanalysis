// The programs of day 43, one per unit:  cargo run -p day43 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day43::*;

/// Day 43: Spectral unmixing. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day43")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Bleed-through and compensation
    #[command(name = "crosstalk")]
    Crosstalk,
    /// Least-squares unmixing
    #[command(name = "lsunmix")]
    Lsunmix,
    /// Endmembers: the geometry of mixtures
    #[command(name = "endmembers")]
    Endmembers,
    /// Stain separation: colour deconvolution
    #[command(name = "stain-unmixing")]
    StainUnmixing,
}

fn main() {
    match Cli::parse().unit {
        Unit::Crosstalk => crosstalk::run::main(),
        Unit::Lsunmix => lsunmix::run::main(),
        Unit::Endmembers => endmembers::run::main(),
        Unit::StainUnmixing => stain_unmixing::run::main(),
    }
}
