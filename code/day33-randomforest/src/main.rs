// The programs of day 33, one per unit:  cargo run -p day33 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day33::*;

/// Day 33: Random forests. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day33")]
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
        Unit::Dtree => dtree::run::main(),
        Unit::Forest => forest::run::main(),
        Unit::ForestObjects => forest_objects::run::main(),
        Unit::Pixelclass => pixelclass::run::main(),
    }
}
