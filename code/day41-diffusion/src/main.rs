// The programs of day 41, one per unit:  cargo run -p day41 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day41::*;

/// Day 41: Diffusion models. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day41")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Noise levels, denoisers and the score
    #[command(name = "score-matching")]
    ScoreMatching,
    /// Samplers, steps, and judging the samples
    #[command(name = "diffusion-sampling")]
    DiffusionSampling,
    /// Conditioning, guidance and latent diffusion
    #[command(name = "guidance")]
    Guidance,
    /// Diffusion priors for inverse problems
    #[command(name = "diffusion-inverse")]
    DiffusionInverse,
}

fn main() {
    match Cli::parse().unit {
        Unit::ScoreMatching => score_matching::run::main(),
        Unit::DiffusionSampling => diffusion_sampling::run::main(),
        Unit::Guidance => guidance::run::main(),
        Unit::DiffusionInverse => diffusion_inverse::run::main(),
    }
}
