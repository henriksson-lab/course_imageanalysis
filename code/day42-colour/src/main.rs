// The programs of day 42, one per unit:  cargo run -p day42 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day42::*;

/// Day 42: Colour. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day42")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// From a spectrum to three numbers
    #[command(name = "colour-vision")]
    ColourVision,
    /// RGB: gamma, linear maps & white balance
    #[command(name = "rgb-linear")]
    RgbLinear,
    /// HSV, CIELAB & colour segmentation
    #[command(name = "colour-perceptual")]
    ColourPerceptual,
    /// Colour for display: maps, merges & decorrelation
    #[command(name = "colour-display")]
    ColourDisplay,
}

fn main() {
    match Cli::parse().unit {
        Unit::ColourVision => colour_vision::run::main(),
        Unit::RgbLinear => rgb_linear::run::main(),
        Unit::ColourPerceptual => colour_perceptual::run::main(),
        Unit::ColourDisplay => colour_display::run::main(),
    }
}
