// The programs of this day, one per unit:  cargo run -p marchingcubes -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Marching cubes. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "marchingcubes")]
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
        Unit::Volumes3d => marchingcubes::volumes3d::run::main(),
        Unit::Isocontour => marchingcubes::isocontour::run::main(),
        Unit::Marchingcubes => marchingcubes::marchingcubes::run::main(),
        Unit::SurfaceMeasure => marchingcubes::surface_measure::run::main(),
    }
}
