# Image analysis: the code

Coding exercises for the image analysis course, in Rust. The course website explains
how to install Rust and use this folder (page "Coding exercises: getting started").

```
imgcourse/           the small image library: Image<T>, Volume<T>, reading and writing files,
                     border rules, complex numbers, random numbers, FFT, small linear algebra,
                     test images (the course micrograph, synthetic nuclei)
starter/             a first program: read an image, change every pixel, save the result
images/        one project per day; one folder per unit in src/, e.g. for "kernels":
  src/kernels/exercise.rs    your code: replace each todo!()
  src/kernels/solution.rs    a model solution
  src/kernels/run.rs         runs your code on real images, pictures in out/kernels/
  tests/check/kernels.rs     checks, using the numbers of the pen-and-paper exercises
edges/ ... gpus/
```

Run everything from this folder (the one with this README):

```
cargo run  -p starter                               # a first program, writes out/starter/
cargo test -p images kernels::                       # check your code for the unit "kernels"
cargo run  -p images -- kernels                      # run it on the micrograph, writes out/kernels/
cargo test -p images --features solution kernels::   # the same checks on the model solution
cargo run  -p images -- --help                       # the units of day 1
cargo doc  -p imgcourse --open                      # the library's documentation in the browser
```

Builds go to `target/`; `cargo clean` removes it.

Conventions of the library: pixel `(x, y)` is column `x` (from the left) and row `y` (from the
top), both counted from 0; grey values are `f32` between 0 (black) and 1 (white). Because `y`
points down, a rotation by a positive angle turns the image **clockwise** on the screen.

The course micrograph (`imgcourse/data/cell256.png`): transmission electron micrograph of a
mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons.
