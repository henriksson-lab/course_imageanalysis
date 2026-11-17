// The programs of day 50, one per unit:  cargo run -p day50 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day50::*;

/// Day 50: Ray tracing. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day50")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Rays from the camera, and where they hit
    #[command(name = "raycast")]
    Raycast,
    /// Shadows, mirrors and glass
    #[command(name = "whitted")]
    Whitted,
    /// Tracing fast: boxes, trees and grids
    #[command(name = "bvh")]
    Bvh,
    /// Path tracing: Monte Carlo and noise
    #[command(name = "pathtrace")]
    Pathtrace,
}

fn main() {
    match Cli::parse().unit {
        Unit::Raycast => raycast::run::main(),
        Unit::Whitted => whitted::run::main(),
        Unit::Bvh => bvh::run::main(),
        Unit::Pathtrace => pathtrace::run::main(),
    }
}
