// The programs of this day, one per unit:  cargo run -p colour -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Colour. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "colour")]
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
        Unit::ColourVision => colour::colour_vision::run::main(),
        Unit::RgbLinear => colour::rgb_linear::run::main(),
        Unit::ColourPerceptual => colour::colour_perceptual::run::main(),
        Unit::ColourDisplay => colour::colour_display::run::main(),
    }
}
