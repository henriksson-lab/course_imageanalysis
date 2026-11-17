// The programs of day 27, one per unit:  cargo run -p day27 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day27::*;

/// Day 27: Graph cuts. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day27")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Segmentation as energy minimisation
    #[command(name = "seg-energy")]
    SegEnergy,
    /// Graphs, cuts and maximum flow
    #[command(name = "maxflow")]
    Maxflow,
    /// Graph-cut segmentation
    #[command(name = "graphcut-seg")]
    GraphcutSeg,
    /// Spectral partitioning and normalized cuts
    #[command(name = "ncut")]
    Ncut,
}

fn main() {
    match Cli::parse().unit {
        Unit::SegEnergy => seg_energy::run::main(),
        Unit::Maxflow => maxflow::run::main(),
        Unit::GraphcutSeg => graphcut_seg::run::main(),
        Unit::Ncut => ncut::run::main(),
    }
}
