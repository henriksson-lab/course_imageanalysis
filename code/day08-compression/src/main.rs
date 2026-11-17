// The programs of day 8, one per unit:  cargo run -p day08 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day08::*;

/// Day 8: Image compression. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day08")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Information, entropy and entropy codes
    #[command(name = "entropy")]
    Entropy,
    /// Prediction, dictionaries and lossless formats
    #[command(name = "lossless")]
    Lossless,
    /// Transform coding, the DCT and the KLT
    #[command(name = "dct")]
    Dct,
    /// The JPEG pipeline, artefacts and quality metrics
    #[command(name = "jpeg")]
    Jpeg,
    /// Compression and measurements, JPEG 2000, formats
    #[command(name = "compress-science")]
    CompressScience,
}

fn main() {
    match Cli::parse().unit {
        Unit::Entropy => entropy::run::main(),
        Unit::Lossless => lossless::run::main(),
        Unit::Dct => dct::run::main(),
        Unit::Jpeg => jpeg::run::main(),
        Unit::CompressScience => compress_science::run::main(),
    }
}
