// The programs of this day, one per unit:  cargo run -p skeleton -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Skeletons. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "skeleton")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names)] // the variants are named after the units
enum Unit {
    /// Hit-or-miss and thinning
    #[command(name = "skel-thinning")]
    SkelThinning,
    /// Medial axis, distance transform and reconstruction
    #[command(name = "skel-medialaxis")]
    SkelMedialaxis,
    /// From skeleton to graph: pruning and length
    #[command(name = "skel-graph")]
    SkelGraph,
    /// Measuring neurons and vessel networks
    #[command(name = "skel-neurovessel")]
    SkelNeurovessel,
}

fn main() {
    match Cli::parse().unit {
        Unit::SkelThinning => skeleton::skel_thinning::run::main(),
        Unit::SkelMedialaxis => skeleton::skel_medialaxis::run::main(),
        Unit::SkelGraph => skeleton::skel_graph::run::main(),
        Unit::SkelNeurovessel => skeleton::skel_neurovessel::run::main(),
    }
}
