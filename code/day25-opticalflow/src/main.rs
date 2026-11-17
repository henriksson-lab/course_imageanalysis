// The programs of day 25, one per unit:  cargo run -p day25 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day25::*;

/// Day 25: Optical flow. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day25")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Motion between frames: brightness constancy and the aperture problem
    #[command(name = "flow-constraint")]
    FlowConstraint,
    /// Lucas–Kanade: local least squares and pyramids
    #[command(name = "lucas-kanade")]
    LucasKanade,
    /// Horn–Schunck and evaluating flow
    #[command(name = "horn-schunck")]
    HornSchunck,
    /// Non-rigid registration
    #[command(name = "nonrigid-reg")]
    NonrigidReg,
}

fn main() {
    match Cli::parse().unit {
        Unit::FlowConstraint => flow_constraint::run::main(),
        Unit::LucasKanade => lucas_kanade::run::main(),
        Unit::HornSchunck => horn_schunck::run::main(),
        Unit::NonrigidReg => nonrigid_reg::run::main(),
    }
}
