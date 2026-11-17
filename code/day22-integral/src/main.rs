// The programs of day 22, one per unit:  cargo run -p day22 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day22::*;

/// Day 22: Integral images. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day22")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Integral images: any rectangle in four lookups
    #[command(name = "integral-image")]
    IntegralImage,
    /// Box filters & local statistics
    #[command(name = "box-filters")]
    BoxFilters,
    /// Rectangle features
    #[command(name = "rect-features")]
    RectFeatures,
    /// Boosted detectors & the cascade
    #[command(name = "viola-jones")]
    ViolaJones,
}

fn main() {
    match Cli::parse().unit {
        Unit::IntegralImage => integral_image::run::main(),
        Unit::BoxFilters => box_filters::run::main(),
        Unit::RectFeatures => rect_features::run::main(),
        Unit::ViolaJones => viola_jones::run::main(),
    }
}
