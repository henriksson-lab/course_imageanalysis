// The programs of day 21, one per unit:  cargo run -p day21 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day21::*;

/// Day 21: Hough transform. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day21")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Hough transform: lines
    #[command(name = "hough-lines")]
    HoughLines,
    /// Hough transform: circles
    #[command(name = "hough-circles")]
    HoughCircles,
    /// The generalised Hough transform
    #[command(name = "ght")]
    Ght,
    /// Hypothesise and verify: RANSAC
    #[command(name = "ransac")]
    Ransac,
}

fn main() {
    match Cli::parse().unit {
        Unit::HoughLines => hough_lines::run::main(),
        Unit::HoughCircles => hough_circles::run::main(),
        Unit::Ght => ght::run::main(),
        Unit::Ransac => ransac::run::main(),
    }
}
