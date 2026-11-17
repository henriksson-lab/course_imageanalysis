// The programs of this day, one per unit:  cargo run -p cellnets -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Networks for cells and objects. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "cellnets")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Star-convex polygons
    #[command(name = "stardist")]
    Stardist,
    /// Flow fields to objects
    #[command(name = "cellpose")]
    Cellpose,
    /// From an image to a list of objects
    #[command(name = "yolo")]
    Yolo,
    /// Promptable segmentation
    #[command(name = "sam")]
    Sam,
    /// Networks and classical pipelines compared
    #[command(name = "cellnets-compare")]
    CellnetsCompare,
}

fn main() {
    match Cli::parse().unit {
        Unit::Stardist => cellnets::stardist::run::main(),
        Unit::Cellpose => cellnets::cellpose::run::main(),
        Unit::Yolo => cellnets::yolo::run::main(),
        Unit::Sam => cellnets::sam::run::main(),
        Unit::CellnetsCompare => cellnets::cellnets_compare::run::main(),
    }
}
