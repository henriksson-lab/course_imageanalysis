// The programs of day 15, one per unit:  cargo run -p day15 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day15::*;

/// Day 15: Watershed. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day15")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names)] // the variants are named after the units
enum Unit {
    /// Images as landscapes: watershed by flooding
    #[command(name = "ws-landscape")]
    WsLandscape,
    /// The flooding algorithm on a grid
    #[command(name = "ws-flooding")]
    WsFlooding,
    /// Watershed segmentation of a real image
    #[command(name = "ws-gradient")]
    WsGradient,
    /// Splitting touching objects with the distance transform
    #[command(name = "ws-distance")]
    WsDistance,
}

fn main() {
    match Cli::parse().unit {
        Unit::WsLandscape => ws_landscape::run::main(),
        Unit::WsFlooding => ws_flooding::run::main(),
        Unit::WsGradient => ws_gradient::run::main(),
        Unit::WsDistance => ws_distance::run::main(),
    }
}
