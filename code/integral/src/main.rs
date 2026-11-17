// The programs of this day, one per unit:  cargo run -p integral -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Integral images. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "integral")]
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
        Unit::IntegralImage => integral::integral_image::run::main(),
        Unit::BoxFilters => integral::box_filters::run::main(),
        Unit::RectFeatures => integral::rect_features::run::main(),
        Unit::ViolaJones => integral::viola_jones::run::main(),
    }
}
