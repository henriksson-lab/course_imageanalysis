// The programs of this day, one per unit:  cargo run -p morphology -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Morphology. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "morphology")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names)] // the variants are named after the units
enum Unit {
    /// Erosion and dilation
    #[command(name = "morph-binary")]
    MorphBinary,
    /// Opening, closing & the distance transform
    #[command(name = "morph-openclose")]
    MorphOpenclose,
    /// Grey-level morphology: top-hat & gradient
    #[command(name = "morph-grey")]
    MorphGrey,
    /// Reconstruction & h-extrema
    #[command(name = "morph-reconstruct")]
    MorphReconstruct,
}

fn main() {
    match Cli::parse().unit {
        Unit::MorphBinary => morphology::morph_binary::run::main(),
        Unit::MorphOpenclose => morphology::morph_openclose::run::main(),
        Unit::MorphGrey => morphology::morph_grey::run::main(),
        Unit::MorphReconstruct => morphology::morph_reconstruct::run::main(),
    }
}
