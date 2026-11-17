// The programs of this day, one per unit:  cargo run -p colocalisation -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Colocalisation. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "colocalisation")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Colocalisation of two channels
    #[command(name = "coloc")]
    Coloc,
    /// Is it more than chance?
    #[command(name = "coloc-null")]
    ColocNull,
    /// Objects instead of pixels
    #[command(name = "coloc-objects")]
    ColocObjects,
    /// What the microscope does to colocalisation
    #[command(name = "coloc-confounds")]
    ColocConfounds,
}

fn main() {
    match Cli::parse().unit {
        Unit::Coloc => colocalisation::coloc::run::main(),
        Unit::ColocNull => colocalisation::coloc_null::run::main(),
        Unit::ColocObjects => colocalisation::coloc_objects::run::main(),
        Unit::ColocConfounds => colocalisation::coloc_confounds::run::main(),
    }
}
