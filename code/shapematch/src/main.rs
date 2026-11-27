// The programs of this day, one per unit:  cargo run -p shapematch -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Shape matching in Fourier space. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "shapematch")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Outlines as periodic signals: Fourier descriptors
    #[command(name = "fourier-descriptors")]
    FourierDescriptors,
    /// Rotation and scale become shifts: Fourier–Mellin
    #[command(name = "fourier-mellin")]
    FourierMellin,
    /// Docking by correlation
    #[command(name = "fft-docking")]
    FftDocking,
}

fn main() {
    match Cli::parse().unit {
        Unit::FourierDescriptors => shapematch::fourier_descriptors::run::main(),
        Unit::FourierMellin => shapematch::fourier_mellin::run::main(),
        Unit::FftDocking => shapematch::fft_docking::run::main(),
    }
}
