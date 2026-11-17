// The programs of day 49, one per unit:  cargo run -p day49 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day49::*;

/// Day 49: Geometry from multiple images. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day49")]
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
        Unit::Cameras => cameras::run::main(),
        Unit::TwoView => two_view::run::main(),
        Unit::Stereo => stereo::run::main(),
        Unit::Sfm => sfm::run::main(),
    }
}
