// The programs of this day, one per unit:  cargo run -p gans -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use gans::*;

/// GANs and normalising flows. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "gans")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Generative adversarial networks
    #[command(name = "gan")]
    Gan,
    /// Training GANs and what goes wrong
    #[command(name = "gan-training")]
    GanTraining,
    /// Image-to-image translation and hallucination
    #[command(name = "image-translation")]
    ImageTranslation,
    /// Normalising flows
    #[command(name = "flows")]
    Flows,
}

fn main() {
    match Cli::parse().unit {
        Unit::Gan => gan::run::main(),
        Unit::GanTraining => gan_training::run::main(),
        Unit::ImageTranslation => image_translation::run::main(),
        Unit::Flows => flows::run::main(),
    }
}
