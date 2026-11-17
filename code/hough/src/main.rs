// The programs of this day, one per unit:  cargo run -p hough -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Hough transform. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "hough")]
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
        Unit::HoughLines => hough::hough_lines::run::main(),
        Unit::HoughCircles => hough::hough_circles::run::main(),
        Unit::Ght => hough::ght::run::main(),
        Unit::Ransac => hough::ransac::run::main(),
    }
}
