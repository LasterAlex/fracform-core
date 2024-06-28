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
use image::{ImageBuffer, Rgb};
use num::Complex;
use rand::distributions::{Alphanumeric, DistString};
use strfmt::Format; // 0.8

pub mod colors;
pub mod config;
pub mod formula;
pub mod fractals;

fn sanitize_filename(name: String) -> String {
    name.replace(" ", "").replace("/", "÷").replace("*", "×")
}

fn create_file_path(formula: &str) -> PathBuf {
    let fractals_path = Path::new(GENERATED_DIR).join(Path::new(FRACTALS_DIR));
    let sanitized_formula = sanitize_filename(formula.to_string());
    let rand_string = Alphanumeric.sample_string(&mut rand::thread_rng(), 8);
    let filename = format!("{}_{}.png", sanitized_formula, rand_string,);
    fractals_path.join(filename.clone())
}

fn save_bitmap(bitmap: &Vec<Vec<(u8, u8, u8)>>, name: &Path) {
    let mut image_buffer =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::new(bitmap.len() as u32, bitmap[0].len() as u32);

    for (x, row) in bitmap.iter().enumerate() {
        for (y, pixel) in row.iter().enumerate() {
            let (r, g, b) = *pixel;
            image_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    image_buffer.save(name).expect("Failed to save image");
}

#[allow(dead_code)]
fn make_animation() {
    // Fractal parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.7;
    let iterations = 1000;
    let palette_mode = PaletteMode::Rainbow { offset: Some(205) };
    let formula = "z * c + z.powf({factor:.2}) + c";

    // Animation parameters
    let start_factor = 0.0;
    let end_factor = 10.0;
    let frame_count = 2000;
    let starting_frame = 0;  // If the animation is interrupted, set this to the last frame + 1

    // Animation generation
    let animation_directory_name = sanitize_filename(
        formula
            .format(&HashMap::from([("factor".to_string(), start_factor)]))
            .unwrap(),
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
            start_factor as f64,
            frame_count as f64,
            end_factor as f64,
        );
        create_formula_project(
            formula
                .format(&HashMap::from([("factor".to_string(), factor)]))
                .unwrap()
                .as_str(),
        )
        .expect("Failed to generate Rust code");
        compile_formula_project().expect("Failed to compile Rust code");
        load_library();

        fractal = Fractal::new(
            width,
            height,
            zoom,
            Complex::new(0.0, 0.0),
            iterations,
            160,
            Some(Complex::new(0.0, 0.0)),
            palette_mode.clone(),
        );
        let bitmap = fractal.clone().mandelbrot();

        let color_bitmap = fractal.make_color_from_bitmap(bitmap);
        let file = current_animation_directory.join(format!("{}_fractal_animated.png", frame));
        save_bitmap(&color_bitmap, file.as_path());
    }
    println!("Animation took {:.2?}", start.elapsed());
}

#[allow(dead_code)]
fn run() {
    // Parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.7;
    let center_coordinates = Complex::new(-0.5, 0.0);
    let iterations = 500;

    // let palette_mode = PaletteMode::Rainbow { offset: None };
    // let palette_mode = PaletteMode::Smooth { shift: None, offset: None };
    // let palette_mode = PaletteMode::BrownAndBlue;
    // let palette_mode = PaletteMode::Custom;
    let palette_mode = PaletteMode::Naive { shift: Some(300), offset: Some(10) };

    // let formula = "z.powc(z - c * z.powf(10.0)) + z.powf(10.0) + c";
    let formula = "z * z + c";
    let fractal_type = FractalType::Mandelbrot;
    let c = Complex::new(0.0, 0.5); // Important only for Julia sets
    let max_abs = 1000;

    // Code
    let start = Instant::now();

    // This exists to make sure that the library is loaded before the formula is generated
    if create_formula_project(&formula).expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();
    println!("Library loaded in {:.2?}", start.elapsed());

    let fractal = Fractal::new(
        width,
        height,
        zoom,
        center_coordinates,
        iterations,
        max_abs, // Not very important unless you know what you are doing
        Some(c),
        palette_mode,
    );

    let fractal_bitmap = match fractal_type {
        FractalType::Mandelbrot => fractal.clone().mandelbrot(),
        FractalType::Julia => fractal.clone().julia(),
        _ => todo!("Add the rest of the types"),
    };
    let color_bitmap = fractal.make_color_from_bitmap(fractal_bitmap);

    let path = create_file_path(formula);
    save_bitmap(
        &color_bitmap,
        &path.as_path()
    );
    println!("Saved with name: {}", path.as_path().display());
}

fn main() {
    let generated_path = Path::new(GENERATED_DIR);
    let animations_path = generated_path.join(Path::new(ANIMATIONS_DIR));
    let fractals_path = generated_path.join(Path::new(FRACTALS_DIR));
    fs::create_dir_all(animations_path).unwrap();
    fs::create_dir_all(fractals_path).unwrap();
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}
