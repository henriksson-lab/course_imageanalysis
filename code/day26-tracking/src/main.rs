// The programs of day 26, one per unit:  cargo run -p day26 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day26::*;

/// Day 26: Tracking. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day26")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names)] // the variants are named after the units
enum Unit {
    /// From detections to tracks
    #[command(name = "track-linking")]
    TrackLinking,
    /// Motion models and the Kalman filter
    #[command(name = "track-predict")]
    TrackPredict,
    /// Gaps, divisions and lineage trees
    #[command(name = "track-events")]
    TrackEvents,
    /// Speed, MSD and modes of motion
    #[command(name = "track-analysis")]
    TrackAnalysis,
}

fn main() {
    match Cli::parse().unit {
        Unit::TrackLinking => track_linking::run::main(),
        Unit::TrackPredict => track_predict::run::main(),
        Unit::TrackEvents => track_events::run::main(),
        Unit::TrackAnalysis => track_analysis::run::main(),
    }
}
