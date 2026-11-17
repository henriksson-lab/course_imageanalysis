// The programs of this day, one per unit:  cargo run -p measurements -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Measurements. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "measurements")]
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
        Unit::IntensityFeatures => measurements::intensity_features::run::main(),
        Unit::TextureFeatures => measurements::texture_features::run::main(),
        Unit::Relations => measurements::relations::run::main(),
    }
}
