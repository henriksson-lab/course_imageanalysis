// The programs of this day, one per unit:  cargo run -p diffusion -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Diffusion models. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "diffusion")]
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
    /// Drifting: generating in one step
    #[command(name = "drifting")]
    Drifting,
}

fn main() {
    match Cli::parse().unit {
        Unit::ScoreMatching => diffusion::score_matching::run::main(),
        Unit::DiffusionSampling => diffusion::diffusion_sampling::run::main(),
        Unit::Guidance => diffusion::guidance::run::main(),
        Unit::DiffusionInverse => diffusion::diffusion_inverse::run::main(),
        Unit::Drifting => diffusion::drifting::run::main(),
    }
}
