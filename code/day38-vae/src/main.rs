// The programs of day 38, one per unit:  cargo run -p day38 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day38::*;

/// Day 38: Variational autoencoders. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day38")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// From PCA to autoencoders
    #[command(name = "autoencoder")]
    Autoencoder,
    /// The variational autoencoder
    #[command(name = "vae")]
    Vae,
    /// Generating and exploring images with a VAE
    #[command(name = "vae-gen")]
    VaeGen,
}

fn main() {
    match Cli::parse().unit {
        Unit::Autoencoder => autoencoder::run::main(),
        Unit::Vae => vae::run::main(),
        Unit::VaeGen => vae_gen::run::main(),
    }
}
