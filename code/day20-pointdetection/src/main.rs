// The programs of day 20, one per unit:  cargo run -p day20 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day20::*;

/// Day 20: Point detection. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day20")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names)] // the variants are named after the units
enum Unit {
    /// Finding spots: filter, then local maxima
    #[command(name = "spot-filters")]
    SpotFilters,
    /// How many false spots? Thresholds from noise
    #[command(name = "spot-threshold")]
    SpotThreshold,
    /// Sub-pixel localisation and the Cramér–Rao bound
    #[command(name = "spot-fit")]
    SpotFit,
    /// Crowded spots and evaluating a detector
    #[command(name = "spot-crowd")]
    SpotCrowd,
}

fn main() {
    match Cli::parse().unit {
        Unit::SpotFilters => spot_filters::run::main(),
        Unit::SpotThreshold => spot_threshold::run::main(),
        Unit::SpotFit => spot_fit::run::main(),
        Unit::SpotCrowd => spot_crowd::run::main(),
    }
}
