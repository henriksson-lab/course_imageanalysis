// The programs of day 19, one per unit:  cargo run -p day19 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day19::*;

/// Day 19: Spatial statistics of objects. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day19")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Point patterns and complete spatial randomness
    #[command(name = "point-patterns")]
    PointPatterns,
    /// Nearest neighbours and empty space: G, F and J
    #[command(name = "nn-functions")]
    NnFunctions,
    /// Ripley's K, the L function and the pair correlation
    #[command(name = "ripley-k")]
    RipleyK,
    /// Is it more than chance? Simulation envelopes
    #[command(name = "envelopes")]
    Envelopes,
    /// Two cell types and an uneven density
    #[command(name = "marked-patterns")]
    MarkedPatterns,
}

fn main() {
    match Cli::parse().unit {
        Unit::PointPatterns => point_patterns::run::main(),
        Unit::NnFunctions => nn_functions::run::main(),
        Unit::RipleyK => ripley_k::run::main(),
        Unit::Envelopes => envelopes::run::main(),
        Unit::MarkedPatterns => marked_patterns::run::main(),
    }
}
