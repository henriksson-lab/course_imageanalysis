// The programs of this day, one per unit:  cargo run -p opticalflow -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Optical flow. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "opticalflow")]
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
        Unit::FlowConstraint => opticalflow::flow_constraint::run::main(),
        Unit::LucasKanade => opticalflow::lucas_kanade::run::main(),
        Unit::HornSchunck => opticalflow::horn_schunck::run::main(),
        Unit::NonrigidReg => opticalflow::nonrigid_reg::run::main(),
    }
}
