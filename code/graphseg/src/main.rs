// The programs of this day, one per unit:  cargo run -p graphseg -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Graph-based segmentation. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "graphseg")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// The random walker
    #[command(name = "random-walker")]
    RandomWalker,
    /// Spanning trees: single linkage and FH
    #[command(name = "mst-seg")]
    MstSeg,
    /// Watershed as a minimum spanning forest
    #[command(name = "ws-msf")]
    WsMsf,
    /// Hierarchical segmentation and merge trees
    #[command(name = "merge-trees")]
    MergeTrees,
    /// Superpixels: SLIC and smaller graphs
    #[command(name = "superpixels")]
    Superpixels,
}

fn main() {
    match Cli::parse().unit {
        Unit::RandomWalker => graphseg::random_walker::run::main(),
        Unit::MstSeg => graphseg::mst_seg::run::main(),
        Unit::WsMsf => graphseg::ws_msf::run::main(),
        Unit::MergeTrees => graphseg::merge_trees::run::main(),
        Unit::Superpixels => graphseg::superpixels::run::main(),
    }
}
