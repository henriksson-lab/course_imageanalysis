// The programs of this day, one per unit:  cargo run -p graphcuts -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Graph cuts. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "graphcuts")]
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
        Unit::SegEnergy => graphcuts::seg_energy::run::main(),
        Unit::Maxflow => graphcuts::maxflow::run::main(),
        Unit::GraphcutSeg => graphcuts::graphcut_seg::run::main(),
        Unit::Ncut => graphcuts::ncut::run::main(),
    }
}
