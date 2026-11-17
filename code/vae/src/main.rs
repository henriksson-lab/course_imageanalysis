// The programs of this day, one per unit:  cargo run -p vae -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Variational autoencoders. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "vae")]
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
        Unit::Autoencoder => vae::autoencoder::run::main(),
        Unit::Vae => vae::vae::run::main(),
        Unit::VaeGen => vae::vae_gen::run::main(),
    }
}
