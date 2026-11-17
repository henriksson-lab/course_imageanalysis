// The programs of day 16, one per unit:  cargo run -p day16 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day16::*;

/// Day 16: Measurements. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day16")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Intensity inside an object
    #[command(name = "intensity-features")]
    IntensityFeatures,
    /// Texture: co-occurrence matrices
    #[command(name = "texture-features")]
    TextureFeatures,
    /// Neighbours, parents and children
    #[command(name = "relations")]
    Relations,
}

fn main() {
    match Cli::parse().unit {
        Unit::IntensityFeatures => intensity_features::run::main(),
        Unit::TextureFeatures => texture_features::run::main(),
        Unit::Relations => relations::run::main(),
    }
}
