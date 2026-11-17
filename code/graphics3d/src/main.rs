// The programs of this day, one per unit:  cargo run -p graphics3d -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// 3D graphics. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "graphics3d")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Meshes, coordinate spaces and projection
    #[command(name = "pipeline3d")]
    Pipeline3d,
    /// From triangles to shaded pixels
    #[command(name = "rasterise")]
    Rasterise,
    /// How graphics cards store colour
    #[command(name = "gpuformats")]
    Gpuformats,
    /// Seeing voxels: projections and volume rendering
    #[command(name = "volrender")]
    Volrender,
}

fn main() {
    match Cli::parse().unit {
        Unit::Pipeline3d => graphics3d::pipeline3d::run::main(),
        Unit::Rasterise => graphics3d::rasterise::run::main(),
        Unit::Gpuformats => graphics3d::gpuformats::run::main(),
        Unit::Volrender => graphics3d::volrender::run::main(),
    }
}
