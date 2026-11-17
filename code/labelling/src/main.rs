// The programs of this day, one per unit:  cargo run -p labelling -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Labelling. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "labelling")]
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
        Unit::LabelConnectivity => labelling::label_connectivity::run::main(),
        Unit::Regionprops => labelling::regionprops::run::main(),
        Unit::ShapeFeatures => labelling::shape_features::run::main(),
        Unit::LabelMeasure => labelling::label_measure::run::main(),
    }
}
