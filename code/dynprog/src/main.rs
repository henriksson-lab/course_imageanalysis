// The programs of this day, one per unit:  cargo run -p dynprog -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Dynamic programming. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "dynprog")]
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
        Unit::DpPaths => dynprog::dp_paths::run::main(),
        Unit::PolarBoundary => dynprog::polar_boundary::run::main(),
        Unit::Livewire => dynprog::livewire::run::main(),
        Unit::Fastmarching => dynprog::fastmarching::run::main(),
    }
}
