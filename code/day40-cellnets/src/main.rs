// The programs of day 40, one per unit:  cargo run -p day40 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day40::*;

/// Day 40: Networks for cells and objects. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day40")]
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
        Unit::Stardist => stardist::run::main(),
        Unit::Cellpose => cellpose::run::main(),
        Unit::Yolo => yolo::run::main(),
        Unit::Sam => sam::run::main(),
        Unit::CellnetsCompare => cellnets_compare::run::main(),
    }
}
