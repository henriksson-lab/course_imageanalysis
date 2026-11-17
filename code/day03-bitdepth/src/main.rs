// The programs of day 3, one per unit:  cargo run -p day03 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day03::*;

/// Day 3: Bit depth and noise. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day03")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Quantisation and bit depth
    #[command(name = "quantisation")]
    Quantisation,
    /// Camera noise, SNR and PSNR
    #[command(name = "cameranoise")]
    Cameranoise,
    /// 12- and 16-bit images in practice
    #[command(name = "highbit")]
    Highbit,
    /// Contrast and intensity transforms
    #[command(name = "contrast")]
    Contrast,
    /// Floating point
    #[command(name = "floatingpoint")]
    Floatingpoint,
}

fn main() {
    match Cli::parse().unit {
        Unit::Quantisation => quantisation::run::main(),
        Unit::Cameranoise => cameranoise::run::main(),
        Unit::Highbit => highbit::run::main(),
        Unit::Contrast => contrast::run::main(),
        Unit::Floatingpoint => floatingpoint::run::main(),
    }
}
