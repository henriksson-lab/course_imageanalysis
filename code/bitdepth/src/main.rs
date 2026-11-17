// The programs of this day, one per unit:  cargo run -p bitdepth -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Bit depth and noise. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "bitdepth")]
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
        Unit::Quantisation => bitdepth::quantisation::run::main(),
        Unit::Cameranoise => bitdepth::cameranoise::run::main(),
        Unit::Highbit => bitdepth::highbit::run::main(),
        Unit::Contrast => bitdepth::contrast::run::main(),
        Unit::Floatingpoint => bitdepth::floatingpoint::run::main(),
    }
}
