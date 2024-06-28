# Fracform Core

This is the core library for fractal generation, named after FRACtal FORMing (cuz fracgen sounds boring)

The goal is to make a library that can very efficiently generate

1) Fractals adjacent to the Mandelbrot set (Mandelbrot set, Julia set, Buddhabrot, Antibuddhabrot, etc)
2) Non-boring other fractals (Newton's fractal, etc)
3) Animations of any of these fractals

I have adored fractals for a very long time, and this is the first actually good and performant attempt

### How to use

#### Creation of the regular fractal:
1) Go to `main.rs` and go down to `fn run()`
2) Tweak the parameters to your liking
3) Ensure that `run` is spawned by `child` in the `fn main()`
4) `cargo run --release`
5) You will find your fractal at ./generated/fractals

#### Creation of animations:
1) Go to `main.rs` and go down to `fn make_animation()`
2) Tweak the parameters to your liking _(`factor:.2` in the formula is a linear function, starting from `start_factor`, and ending on `end_factor`)_
3) Ensure that `make_animation` is spawned by `child` in the `fn main()`
4) `cargo run --release`
5) You will find your animation frames at ./generated/animations/_this\_changes\_for\_every\_formula_
6) Make it into an mp4 (porting the script for that to rust is ongoing)
7) You will find your animation mp4  at ./generated/animations

#### config.rs:

`MAX_PIXELS` - the max amount of pixels that can be generated (needs to be known at compile time)
`JOBS` - the jobs created to generate fractals. In general, if the fractal is small and fast - 1 job is better, cuz spawning jobs costs time, but if it's a big and complicated fractal that takes seconds to make, tweaking `JOBS` will improve the speed
`STACK_SIZE` - the size of the stack. The bigger it is, the bigger `MAX_PIXELS` can get, but also slower startup times spent on allocating that stack
`*_DIR` - The name of the directories, NOT PATH! The path structure is set in code for consistency

#### Notes:

1) The first compilation of the project will be slower than the rest, because it will build `formula_project` in the project directory to allow for dynamic formulas

2) Palettes are not yet very refined, but i very much plan on changing it! These are the current ones:

##### Rainbow palette:
![rainbow palette](https://github.com/LasterAlex/fracform-core/blob/master/fractal_examples/rainbow_palette.png?raw=true)

##### Smooth palette:
![smooth palette](https://github.com/LasterAlex/fracform-core/blob/master/fractal_examples/smooth_palette.png?raw=true)

##### Brown and blue palette:
![brown and blue palette](https://github.com/LasterAlex/fracform-core/blob/master/fractal_examples/brown_and_blue_palette.png?raw=true)

##### Naive palette (shift = 300):
![naive palette with shift 300](https://github.com/LasterAlex/fracform-core/blob/master/fractal_examples/naive_palette.png?raw=true)

### Todo list:
- [x] Add mandelbrot and julia generation
- [x] Parallelize the code
- [x] Add different colorings
- [x] Animation generation
- [ ] Add buddhabrot and antibuddhabrot
- [ ] Make the GOOD coloring to all of fractals
- [ ] Add nebulabrot
- [ ] Add a smart way of memorizing every fractal (save every parameter to recreate the fractal)
- [ ] Make some form of CLI
- [ ] Convert it to an actual library, instead of a regular project 
