// The programs of day 32, one per unit:  cargo run -p day32 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day32::*;

/// Day 32: Dynamic programming. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day32")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// The DP principle: minimal paths on a grid
    #[command(name = "dp-paths")]
    DpPaths,
    /// Closed boundaries: DP in polar coordinates
    #[command(name = "polar-boundary")]
    PolarBoundary,
    /// When there is no order: Dijkstra and live-wire
    #[command(name = "livewire")]
    Livewire,
    /// From Dijkstra to fast marching
    #[command(name = "fastmarching")]
    Fastmarching,
}

fn main() {
    match Cli::parse().unit {
        Unit::DpPaths => dp_paths::run::main(),
        Unit::PolarBoundary => polar_boundary::run::main(),
        Unit::Livewire => livewire::run::main(),
        Unit::Fastmarching => fastmarching::run::main(),
    }
}
