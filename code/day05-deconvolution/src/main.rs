// The programs of day 5, one per unit:  cargo run -p day05 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day05::*;

/// Day 5: Deconvolution. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day05")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Linear deconvolution: inverse, Wiener, Tikhonov
    #[command(name = "deconv")]
    Deconv,
    /// Regularisation: what we add, and how much
    #[command(name = "regularise")]
    Regularise,
    /// Richardson–Lucy, 3D, and the PSF
    #[command(name = "richardson-lucy")]
    RichardsonLucy,
    /// Denoising and what you may measure
    #[command(name = "denoise")]
    Denoise,
}

fn main() {
    match Cli::parse().unit {
        Unit::Deconv => deconv::run::main(),
        Unit::Regularise => regularise::run::main(),
        Unit::RichardsonLucy => richardson_lucy::run::main(),
        Unit::Denoise => denoise::run::main(),
    }
}
