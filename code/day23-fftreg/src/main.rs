// The programs of day 23, one per unit:  cargo run -p day23 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day23::*;

/// Day 23: FFT registration. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day23")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Finding a shift by sliding and scoring
    #[command(name = "matching")]
    Matching,
    /// Cross-correlation with the FFT
    #[command(name = "xcorr-fft")]
    XcorrFft,
    /// Phase correlation
    #[command(name = "phasecorr")]
    Phasecorr,
    /// From a peak to a mosaic
    #[command(name = "stitching")]
    Stitching,
}

fn main() {
    match Cli::parse().unit {
        Unit::Matching => matching::run::main(),
        Unit::XcorrFft => xcorr_fft::run::main(),
        Unit::Phasecorr => phasecorr::run::main(),
        Unit::Stitching => stitching::run::main(),
    }
}
