// The programs of this day, one per unit:  cargo run -p watershed -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Watershed. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "watershed")]
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
        Unit::WsLandscape => watershed::ws_landscape::run::main(),
        Unit::WsFlooding => watershed::ws_flooding::run::main(),
        Unit::WsGradient => watershed::ws_gradient::run::main(),
        Unit::WsDistance => watershed::ws_distance::run::main(),
    }
}
