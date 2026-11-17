// The programs of this day, one per unit:  cargo run -p fftreg -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// FFT registration. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "fftreg")]
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
        Unit::Matching => fftreg::matching::run::main(),
        Unit::XcorrFft => fftreg::xcorr_fft::run::main(),
        Unit::Phasecorr => fftreg::phasecorr::run::main(),
        Unit::Stitching => fftreg::stitching::run::main(),
    }
}
