// The programs of day 45, one per unit:  cargo run -p day45 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day45::*;

/// Day 45: FRET. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day45")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// FRET in pictures: a ruler for nanometres
    #[command(name = "fret-physics")]
    FretPhysics,
    /// Three images, two controls: sensitised emission
    #[command(name = "sensitised-emission")]
    SensitisedEmission,
    /// Acceptor photobleaching, and what an efficiency means
    #[command(name = "acceptor-bleach")]
    AcceptorBleach,
    /// Ratio imaging, biosensors and the noise of a ratio
    #[command(name = "ratiometric")]
    Ratiometric,
}

fn main() {
    match Cli::parse().unit {
        Unit::FretPhysics => fret_physics::run::main(),
        Unit::SensitisedEmission => sensitised_emission::run::main(),
        Unit::AcceptorBleach => acceptor_bleach::run::main(),
        Unit::Ratiometric => ratiometric::run::main(),
    }
}
