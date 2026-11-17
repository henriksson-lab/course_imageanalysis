// The programs of this day, one per unit:  cargo run -p snakes -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Snakes and active shape models. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "snakes")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Active contours (snakes)
    #[command(name = "snakes")]
    Snakes,
    /// The snake as a linear system
    #[command(name = "snake-matrix")]
    SnakeMatrix,
    /// Shape models: landmarks, Procrustes and PCA
    #[command(name = "shape-models")]
    ShapeModels,
    /// Active shape models
    #[command(name = "asm")]
    Asm,
}

fn main() {
    match Cli::parse().unit {
        Unit::Snakes => snakes::snakes::run::main(),
        Unit::SnakeMatrix => snakes::snake_matrix::run::main(),
        Unit::ShapeModels => snakes::shape_models::run::main(),
        Unit::Asm => snakes::asm::run::main(),
    }
}
