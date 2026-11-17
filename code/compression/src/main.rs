// The programs of this day, one per unit:  cargo run -p compression -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Image compression. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "compression")]
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
        Unit::Entropy => compression::entropy::run::main(),
        Unit::Lossless => compression::lossless::run::main(),
        Unit::Dct => compression::dct::run::main(),
        Unit::Jpeg => compression::jpeg::run::main(),
        Unit::CompressScience => compression::compress_science::run::main(),
    }
}
