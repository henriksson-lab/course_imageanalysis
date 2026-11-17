// The programs of this day, one per unit:  cargo run -p ising -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Ising model and Markov random fields. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "ising")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Energy and probability: the Ising model
    #[command(name = "ising")]
    Ising,
    /// Sampling, temperature and the phase transition
    #[command(name = "mcmc")]
    Mcmc,
    /// Bayesian denoising with an Ising prior
    #[command(name = "mrf-denoise")]
    MrfDenoise,
    /// Quadratic smoothing and total variation
    #[command(name = "gmrf")]
    Gmrf,
}

fn main() {
    match Cli::parse().unit {
        Unit::Ising => ising::ising::run::main(),
        Unit::Mcmc => ising::mcmc::run::main(),
        Unit::MrfDenoise => ising::mrf_denoise::run::main(),
        Unit::Gmrf => ising::gmrf::run::main(),
    }
}
