// The programs of this day, one per unit:  cargo run -p multiview -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Geometry from multiple images. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "multiview")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// The pinhole camera and calibration
    #[command(name = "cameras")]
    Cameras,
    /// Two views: epipolar geometry, triangulation
    #[command(name = "two-view")]
    TwoView,
    /// Stereo: from disparity to depth
    #[command(name = "stereo")]
    Stereo,
    /// Structure from motion and bundle adjustment
    #[command(name = "sfm")]
    Sfm,
}

fn main() {
    match Cli::parse().unit {
        Unit::Cameras => multiview::cameras::run::main(),
        Unit::TwoView => multiview::two_view::run::main(),
        Unit::Stereo => multiview::stereo::run::main(),
        Unit::Sfm => multiview::sfm::run::main(),
    }
}
