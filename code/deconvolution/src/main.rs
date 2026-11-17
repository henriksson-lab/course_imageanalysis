// The programs of this day, one per unit:  cargo run -p deconvolution -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Deconvolution. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "deconvolution")]
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
        Unit::Deconv => deconvolution::deconv::run::main(),
        Unit::Regularise => deconvolution::regularise::run::main(),
        Unit::RichardsonLucy => deconvolution::richardson_lucy::run::main(),
        Unit::Denoise => deconvolution::denoise::run::main(),
    }
}
