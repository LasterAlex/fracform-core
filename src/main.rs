#![allow(clippy::too_many_arguments)]
#![allow(clippy::declare_interior_mutable_const)]
#![allow(static_mut_refs)]
#![allow(clippy::borrow_interior_mutable_const)]
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use colors::PaletteMode;
use config::{ANIMATIONS_DIR, FRACTALS_DIR, GENERATED_DIR, STACK_SIZE};
use formula::{compile_formula_project, create_formula_project, load_library};
use fractals::{f, Fractal, FractalType};
use image::{imageops, GenericImageView, ImageBuffer, Rgb};
use num::Complex;
use rand::distr::{Alphanumeric, SampleString};
use strfmt::Format;

use crate::{random_formula::get_random_formula, train::train};

pub mod colors;
pub mod compare_shadows;
pub mod config;
pub mod formula;
pub mod fractals;
pub mod frames_to_mp4;
pub mod random_formula;
pub mod train;
pub mod ui;

fn sanitize_filename(name: String) -> String {
    let mut sanitized_formula = name.replace(" ", "").replace("/", "÷").replace("*", "×");
    sanitized_formula.truncate(150);
    sanitized_formula
}

fn create_file_path(formula: &str) -> PathBuf {
    let fractals_path = Path::new(GENERATED_DIR).join(Path::new(FRACTALS_DIR));
    let sanitized_formula = sanitize_filename(formula.to_string());
    let rand_string = Alphanumeric.sample_string(&mut rand::rng(), 8);
    let filename = format!("{sanitized_formula}_{rand_string}.png");
    fractals_path.join(filename.clone())
}

fn save_bitmap(bitmap: &[Vec<(u8, u8, u8)>], name: &Path, rotate: bool) {
    let mut image_buffer =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::new(bitmap.len() as u32, bitmap[0].len() as u32);

    for (x, row) in bitmap.iter().enumerate() {
        for (y, pixel) in row.iter().enumerate() {
            let (r, g, b) = *pixel;
            image_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    if rotate {
        image_buffer = imageops::rotate270(&image_buffer);
    }

    image_buffer.save(name).expect("Failed to save image");
}

fn make_fractal(fractal: &mut Fractal, fractal_type: FractalType) -> Vec<Vec<(u8, u8, u8)>> {
    if let FractalType::Nebulabrot {
        rounds,
        red_iters,
        green_iters,
        blue_iters,
        color_shift,
        uniform_factor,
    } = fractal_type
    {
        fractal.nebulabrot(
            rounds,
            red_iters,
            green_iters,
            blue_iters,
            color_shift,
            uniform_factor,
        )
    } else if let FractalType::Antinebulabrot {
        rounds,
        red_iters,
        green_iters,
        blue_iters,
        color_shift,
        uniform_factor,
    } = fractal_type
    {
        fractal.antinebulabrot(
            rounds,
            red_iters,
            green_iters,
            blue_iters,
            color_shift,
            uniform_factor,
        )
    } else {
        let fractal_bitmap = match fractal_type {
            FractalType::Mandelbrot => fractal.mandelbrot(),
            FractalType::Julia => fractal.julia(),
            FractalType::Buddhabrot { rounds } => fractal.buddhabrot(rounds),
            FractalType::Antibuddhabrot { rounds } => fractal.antibuddhabrot(rounds),
            _ => {
                unreachable!()
            }
        };
        fractal.make_color_from_bitmap(fractal_bitmap)
    }
}

fn make_and_save_fractal(
    fractal: &mut Fractal,
    fractal_type: FractalType,
    file_path: &Path,
    rotate: bool,
) {
    let color_bitmap = make_fractal(fractal, fractal_type);
    save_bitmap(&color_bitmap, file_path, rotate);
}

#[allow(dead_code)]
fn make_animation() {
    // Fractal parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.25;
    let iterations = 100;
    let max_abs = 32;
    let center_coordinates = Complex::new(0.0, 0.0);
    // let palette_mode = PaletteMode::Smooth {
    //     shift: Some(50),
    //     offset: None,
    // };
    // let palette_mode = PaletteMode::GrayScale {
    //     shift: None,
    //     uniform_factor: Some(0.5),
    // };
    let palette_mode = PaletteMode::Rainbow { offset: Some(100) };
    let formula =
        "((-3.839-z/c)/((c-z)/(-0.238+z)+z)) * {factor:.6} + (z * z + c) * (1 - {factor:.6})";
    let additional_info = "";
    // let fractal_type = FractalType::Nebulabrot {
    //     rounds: 100_000_000,
    //     red_iters: 1000,
    //     green_iters: 100,
    //     blue_iters: 10,
    //     color_shift: None,
    //     uniform_factor: Some(0.9),
    // };
    let fractal_type = FractalType::Mandelbrot;
    // let fractal_type = FractalType::Antibuddhabrot { rounds: 20_000_000 };

    // Animation parameters
    let start_factor = 0.0;
    let end_factor = 1.4;
    let frame_count = 700;
    let starting_frame = 0; // If the animation is interrupted, set this to the last frame + 1
                            // Set to frame_count + 1 if you want to tweak the fps
    let fps = 30;

    // Animation generation
    let animation_directory_name = sanitize_filename(
        formula
            .format(&HashMap::from([("factor".to_string(), start_factor)]))
            .unwrap()
            + additional_info,
    );
    let generated_path = Path::new(GENERATED_DIR);
    let animations_path = generated_path.join(Path::new(ANIMATIONS_DIR));
    let current_animation_directory = animations_path.join(animation_directory_name);

    fs::create_dir_all(current_animation_directory.as_path()).unwrap();

    let mut factor;
    let mut fractal;
    let start = Instant::now();
    for frame in starting_frame..=frame_count {
        println!("{:.2?}%", frame as f64 / frame_count as f64 * 100.0);
        factor = f(
            frame as f64,
            0.0,
            start_factor,
            frame_count as f64,
            end_factor,
        );
        println!("factor: {factor}");
        create_formula_project(
            formula
                .format(&HashMap::from([("factor".to_string(), factor)]))
                .unwrap()
                .as_str(),
        )
        .expect("Failed to generate Rust code");
        println!(
            "Formula {:?}",
            formula
                .format(&HashMap::from([("factor".to_string(), factor)]))
                .unwrap()
                .as_str(),
        );
        compile_formula_project().expect("Failed to compile Rust code");
        load_library();

        fractal = Fractal::new(
            width,
            height,
            zoom,
            center_coordinates,
            iterations,
            max_abs,
            Some(Complex::new(0.0, 0.0)),
            palette_mode.clone(),
        );
        let file = current_animation_directory.join(format!("{frame}_fractal_animated.png"));
        make_and_save_fractal(&mut fractal, fractal_type.clone(), &file, false);
    }
    println!("Animation took {:.2?}", start.elapsed());
    frames_to_mp4::make_mp4(current_animation_directory.as_path(), fps);
}

pub fn get_img_dimensions(filename: &str) -> (i32, i32) {
    let img = image::open(filename).unwrap();
    let (w, h) = img.dimensions();
    (w as i32, h as i32)
}

#[allow(dead_code)]
fn train_fractal() {
    let train_filename = "pssy.jpg";

    let (mut width, mut height) = get_img_dimensions(train_filename);
    // Parameters
    let zoom = 0.5;
    let center_coordinates = Complex::new(0.0, 0.0);
    let iterations = 101;

    let palette_mode = PaletteMode::Rainbow { offset: Some(100) };

    let (formula, rotate) = &train(train_filename);

    if *rotate {
        let tmp = width;
        width = height;
        height = tmp;
    }

    // let formula = "pow((pow(-0.039, imag(asin(c))) / 0.535 + pow(0.831 * ((sqrt(z * (z - -0.617) + c + -1.534) * -0.862 * c + sinh(c)) * c + c + c) - (imag(z + 1.493) - c * c) - c, 1.052) + 3.661 + -3.994 + 0.099) / 1.109 + (c + imag(c + -2.094)) * c - 0.251 * c + -0.521 - -0.543, pow(0.999, z * 1.637)) / 0.952 + -2.978 - -0.377 - -2.939";
    // let formula = "(asin(pow((((sinh((z * c * z + atan(c - sinh(z) * 1.61)) / (sqrt(c) + z)) * 1.03 * (c + -0.02) / c) / c) / c) * (c - real(c)) * c * (c + z) / ((pow(c, c * 0.883) / 9.298) / 0.094) + z, 0.978 + 0.724)) * 1.265 / c) * c";
    println!("{}", formula);
    let fractal_type = FractalType::Mandelbrot;

    let c = Complex::new(0.0, 0.0); // Important only for Julia sets
    let max_abs = 1280;

    // Code
    let start = Instant::now();

    // This exists to make sure that the library is loaded before the formula is generated
    if create_formula_project(formula).expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();
    println!("Library loaded in {:.2?}", start.elapsed());

    let mut fractal = Fractal::new(
        width,
        height,
        zoom,
        center_coordinates,
        iterations,
        max_abs, // Not very important unless you know what you are doing
        Some(c),
        palette_mode,
    );

    let path = create_file_path(formula);
    make_and_save_fractal(&mut fractal, fractal_type, &path, *rotate);
    println!("Saved with name: {}", path.as_path().display());
}

#[allow(dead_code)]
fn run() {
    // Parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.25;
    let center_coordinates = Complex::new(0.0, 0.0);
    let iterations = 100;

    // let palette_mode = PaletteMode::GrayScale {
    //     shift: None,
    //     uniform_factor: Some(0.5),
    // };
    let palette_mode = PaletteMode::Rainbow { offset: Some(100) };
    // let palette_mode = PaletteMode::BrownAndBlue;
    // let palette_mode = PaletteMode::Smooth {
    //     shift: Some(50),
    //     offset: None,
    // };

    // let formula = &get_random_formula();
    // let formula = "(sinh(asin(z) * z) + (c + real(c) + 0.006) * pow(c + z, c) + (z / (4.01 - z)) * tan(atanh(c)) * (z * z * z * z + c * c) * (c + z - -0.475) + imag(tanh(z))) / pow(1.054, c - z + c - -1.216) + atanh(tan(pow(-0.07, asin(pow(5.004 + -0.635, imag(z))))))";
    let formula = "(-3.839-z/c)/((c-z)/(-0.238+z)+z)";

    // let fractal_type = FractalType::Antinebulabrot {
    //     rounds: 100_000_000,
    //     red_iters: 5000,
    //     green_iters: 500,
    //     blue_iters: 50,
    //     color_shift: None,
    //     uniform_factor: Some(0.5),
    // };
    // let fractal_type = FractalType::Buddhabrot {
    //     rounds: width * height * 4_u32.pow(2),
    // };
    // let fractal_type = FractalType::Antibuddhabrot { rounds: 20_000_000 };
    // let fractal_type = FractalType::Buddhabrot { rounds: 400_000_000 };
    let fractal_type = FractalType::Mandelbrot;
    // let fractal_type = FractalType::Julia;
    let c = Complex::new(0.0, 0.0); // Important only for Julia sets
    let max_abs = 32;

    // Code
    let start = Instant::now();

    // This exists to make sure that the library is loaded before the formula is generated
    if create_formula_project(formula).expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();
    println!("Library loaded in {:.2?}", start.elapsed());

    let mut fractal = Fractal::new(
        width,
        height,
        zoom,
        center_coordinates,
        iterations,
        max_abs, // Not very important unless you know what you are doing
        Some(c),
        palette_mode,
    );

    let path = create_file_path(formula);
    make_and_save_fractal(&mut fractal, fractal_type, &path, false);
    println!("Saved with name: {}", path.as_path().display());
}

fn main() {
    let generated_path = Path::new(GENERATED_DIR);
    let animations_path = generated_path.join(Path::new(ANIMATIONS_DIR));
    let fractals_path = generated_path.join(Path::new(FRACTALS_DIR));
    fs::create_dir_all(animations_path).unwrap();
    fs::create_dir_all(fractals_path).unwrap();
    // let child = thread::Builder::new()
    //     .stack_size(STACK_SIZE)
    //     .spawn(run)
    //     .unwrap();
    //
    // // Wait for thread to join
    // child.join().unwrap();
    ui::run_app();
}
