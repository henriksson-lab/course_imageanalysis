// The programs of day 28, one per unit:  cargo run -p day28 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day28::*;

/// Day 28: Ising model and Markov random fields. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day28")]
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
        Unit::Ising => ising::run::main(),
        Unit::Mcmc => mcmc::run::main(),
        Unit::MrfDenoise => mrf_denoise::run::main(),
        Unit::Gmrf => gmrf::run::main(),
    }
}
