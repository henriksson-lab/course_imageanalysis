// The programs of this day, one per unit:  cargo run -p raytracing -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Ray tracing. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "raytracing")]
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
        Unit::Raycast => raytracing::raycast::run::main(),
        Unit::Whitted => raytracing::whitted::run::main(),
        Unit::Bvh => raytracing::bvh::run::main(),
        Unit::Pathtrace => raytracing::pathtrace::run::main(),
    }
}
