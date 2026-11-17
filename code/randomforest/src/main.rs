// The programs of this day, one per unit:  cargo run -p randomforest -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Random forests. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "randomforest")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Decision trees
    #[command(name = "dtree")]
    Dtree,
    /// Bagging and random forests
    #[command(name = "forest")]
    Forest,
    /// Feature tables & feature importance
    #[command(name = "forest-objects")]
    ForestObjects,
    /// Pixel classification with a forest
    #[command(name = "pixelclass")]
    Pixelclass,
}

fn main() {
    match Cli::parse().unit {
        Unit::Dtree => randomforest::dtree::run::main(),
        Unit::Forest => randomforest::forest::run::main(),
        Unit::ForestObjects => randomforest::forest_objects::run::main(),
        Unit::Pixelclass => randomforest::pixelclass::run::main(),
    }
}
