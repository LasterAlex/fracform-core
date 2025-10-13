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
use strfmt::Format;

pub mod colors;
pub mod config;
pub mod formula;
pub mod fractals;
pub mod frames_to_mp4;

fn sanitize_filename(name: String) -> String {
    name.replace(" ", "").replace("/", "÷").replace("*", "×")
}

fn create_file_path(formula: &str) -> PathBuf {
    let fractals_path = Path::new(GENERATED_DIR).join(Path::new(FRACTALS_DIR));
    let sanitized_formula = sanitize_filename(formula.to_string());
    let rand_string = Alphanumeric.sample_string(&mut rand::thread_rng(), 8);
    let filename = format!("{sanitized_formula}_{rand_string}.png");
    fractals_path.join(filename.clone())
}

fn save_bitmap(bitmap: &[Vec<(u8, u8, u8)>], name: &Path) {
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

fn make_and_save_fractal(fractal: &mut Fractal, fractal_type: FractalType, file_path: &Path) {
    let color_bitmap;
    if let FractalType::Nebulabrot {
        rounds,
        red_iters,
        green_iters,
        blue_iters,
        color_shift,
        uniform_factor,
    } = fractal_type
    {
        color_bitmap = fractal.nebulabrot(
            rounds,
            red_iters,
            green_iters,
            blue_iters,
            color_shift,
            uniform_factor,
        );
    } else if let FractalType::Antinebulabrot {
        rounds,
        red_iters,
        green_iters,
        blue_iters,
        color_shift,
        uniform_factor,
    } = fractal_type
    {
        color_bitmap = fractal.antinebulabrot(
            rounds,
            red_iters,
            green_iters,
            blue_iters,
            color_shift,
            uniform_factor,
        );
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
        color_bitmap = fractal.make_color_from_bitmap(fractal_bitmap);
    }
    save_bitmap(&color_bitmap, file_path);
}

#[allow(dead_code)]
fn make_animation() {
    // Fractal parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.2;
    let iterations = 1000;
    let max_abs = 1280;
    // let palette_mode = PaletteMode::Smooth {
    //     shift: Some(200),
    //     offset: None,
    // };
    let palette_mode = PaletteMode::Rainbow { offset: Some(100) };
    let formula = "(z.sinh() + c + 1.0) / 2.0 + c / (z + {factor:.6})";
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

    // Animation parameters
    let start_factor = 0.0;
    let end_factor = 1.5;
    let frame_count = 150;
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
            Complex::new(0.0, 0.0),
            iterations,
            max_abs,
            None,
            palette_mode.clone(),
        );
        let file = current_animation_directory.join(format!("{frame}_fractal_animated.png"));
        make_and_save_fractal(&mut fractal, fractal_type.clone(), &file);
    }
    println!("Animation took {:.2?}", start.elapsed());
    frames_to_mp4::make_mp4(current_animation_directory.as_path(), fps);
}

#[allow(dead_code)]
fn run() {
    // Parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.5;
    let center_coordinates = Complex::new(0.0, 0.0);
    let iterations = 1000;

    // let palette_mode = PaletteMode::GrayScale {
    //     shift: None,
    //     uniform_factor: Some(0.5),
    // };
    let palette_mode = PaletteMode::Rainbow { offset: Some(100) };
    // let palette_mode = PaletteMode::BrownAndBlue;
    // let palette_mode = PaletteMode::Smooth {
    //     shift: Some(200),
    //     offset: None,
    // };

    let formula = "z*z + c";
    // let fractal_type = FractalType::Nebulabrot {
    //     rounds: 100_000_000,
    //     red_iters: 5000,
    //     green_iters: 500,
    //     blue_iters: 50,
    //     color_shift: None,
    //     uniform_factor: Some(0.9),
    // };
    // let fractal_type = FractalType::Buddhabrot {
    //     rounds: width * height * 4_u32.pow(2),
    // };
    // let fractal_type = FractalType::Antibuddhabrot { rounds: 20_000_000 };
    // let fractal_type = FractalType::Buddhabrot { rounds: 400_000_000 };
    let fractal_type = FractalType::Mandelbrot;
    // let fractal_type = FractalType::Julia;
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
    make_and_save_fractal(&mut fractal, fractal_type, &path);
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
