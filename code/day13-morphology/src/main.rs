// The programs of day 13, one per unit:  cargo run -p day13 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day13::*;

/// Day 13: Morphology. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day13")]
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
        Unit::MorphBinary => morph_binary::run::main(),
        Unit::MorphOpenclose => morph_openclose::run::main(),
        Unit::MorphGrey => morph_grey::run::main(),
        Unit::MorphReconstruct => morph_reconstruct::run::main(),
    }
}
