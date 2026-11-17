// The programs of day 39, one per unit:  cargo run -p day39 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day39::*;

/// Day 39: Modern deep learning. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day39")]
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
        Unit::Instances => instances::run::main(),
        Unit::Detection => detection::run::main(),
        Unit::Attention => attention::run::main(),
        Unit::Diffusion => diffusion::run::main(),
    }
}
