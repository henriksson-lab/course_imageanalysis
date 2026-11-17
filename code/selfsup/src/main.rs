// The programs of this day, one per unit:  cargo run -p selfsup -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use selfsup::*;

/// Learning without labels. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "selfsup")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Pretext tasks and masked image modelling
    #[command(name = "pretext")]
    Pretext,
    /// Contrastive learning
    #[command(name = "contrastive")]
    Contrastive,
    /// Self-distillation and foundation models
    #[command(name = "self-distillation")]
    SelfDistillation,
    /// Self-supervised denoising
    #[command(name = "self-denoise")]
    SelfDenoise,
}

fn main() {
    match Cli::parse().unit {
        Unit::Pretext => pretext::run::main(),
        Unit::Contrastive => contrastive::run::main(),
        Unit::SelfDistillation => self_distillation::run::main(),
        Unit::SelfDenoise => self_denoise::run::main(),
    }
}
