// The programs of this day, one per unit:  cargo run -p unmixing -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Spectral unmixing. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "unmixing")]
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
        Unit::Crosstalk => unmixing::crosstalk::run::main(),
        Unit::Lsunmix => unmixing::lsunmix::run::main(),
        Unit::Endmembers => unmixing::endmembers::run::main(),
        Unit::StainUnmixing => unmixing::stain_unmixing::run::main(),
    }
}
