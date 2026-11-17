// The programs of day 47, one per unit:  cargo run -p day47 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day47::*;

/// Day 47: Marching cubes. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day47")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// 3D images
    #[command(name = "volumes3d")]
    Volumes3d,
    /// Level sets & marching squares
    #[command(name = "isocontour")]
    Isocontour,
    /// Marching cubes
    #[command(name = "marchingcubes")]
    Marchingcubes,
    /// Measuring surfaces
    #[command(name = "surface-measure")]
    SurfaceMeasure,
}

fn main() {
    match Cli::parse().unit {
        Unit::Volumes3d => volumes3d::run::main(),
        Unit::Isocontour => isocontour::run::main(),
        Unit::Marchingcubes => marchingcubes::run::main(),
        Unit::SurfaceMeasure => surface_measure::run::main(),
    }
}
