// The programs of day 14, one per unit:  cargo run -p day14 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day14::*;

/// Day 14: Labelling. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day14")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Connectivity and labelling
    #[command(name = "label-connectivity")]
    LabelConnectivity,
    /// Region properties and moments
    #[command(name = "regionprops")]
    Regionprops,
    /// Shape features
    #[command(name = "shape-features")]
    ShapeFeatures,
    /// Working with label images
    #[command(name = "label-measure")]
    LabelMeasure,
}

fn main() {
    match Cli::parse().unit {
        Unit::LabelConnectivity => label_connectivity::run::main(),
        Unit::Regionprops => regionprops::run::main(),
        Unit::ShapeFeatures => shape_features::run::main(),
        Unit::LabelMeasure => label_measure::run::main(),
    }
}
