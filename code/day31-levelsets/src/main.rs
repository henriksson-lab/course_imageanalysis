// The programs of day 31, one per unit:  cargo run -p day31 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day31::*;

/// Day 31: Level sets. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day31")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Curves as level sets
    #[command(name = "implicit")]
    Implicit,
    /// Moving a curve: the level set equation
    #[command(name = "levelset-eq")]
    LevelsetEq,
    /// Edge-based segmentation: geodesic active contours
    #[command(name = "geodesic")]
    Geodesic,
    /// Region-based segmentation: Chan–Vese
    #[command(name = "chanvese")]
    Chanvese,
}

fn main() {
    match Cli::parse().unit {
        Unit::Implicit => implicit::run::main(),
        Unit::LevelsetEq => levelset_eq::run::main(),
        Unit::Geodesic => geodesic::run::main(),
        Unit::Chanvese => chanvese::run::main(),
    }
}
