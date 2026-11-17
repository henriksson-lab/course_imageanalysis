// The programs of this day, one per unit:  cargo run -p moderndl -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Modern deep learning. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "moderndl")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// From masks to objects
    #[command(name = "instances")]
    Instances,
    /// Finding objects as boxes
    #[command(name = "detection")]
    Detection,
    /// Attention, transformers and pretrained models
    #[command(name = "attention")]
    Attention,
    /// Diffusion models and generated detail
    #[command(name = "diffusion")]
    Diffusion,
}

fn main() {
    match Cli::parse().unit {
        Unit::Instances => moderndl::instances::run::main(),
        Unit::Detection => moderndl::detection::run::main(),
        Unit::Attention => moderndl::attention::run::main(),
        Unit::Diffusion => moderndl::diffusion::run::main(),
    }
}
