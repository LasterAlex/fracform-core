use std::{thread, time::Instant};

use colors::PaletteMode;
use config::{JOBS, STACK_SIZE};
use formula::{compile_formula_project, create_formula_project, load_library};
use fractals::{f, Fractal};
use image::{ImageBuffer, Rgb};
use num::Complex;

pub mod colors;
pub mod config;
pub mod formula;
pub mod fractals;

fn save_bitmap(bitmap: &Vec<Vec<(u8, u8, u8)>>, name: &str) {
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

fn make_animation() {
    // Fractal parameters
    let width = 1000;
    let height = 1000;
    let zoom = 0.7;
    let iterations = 1000;
    let palette_mode = PaletteMode::Rainbow { offset: Some(205) };

    // Animation parameters
    let start_factor = 10.0;
    let end_factor = 0.0;
    let frame_count = 2000;
    let animation_foulder =
        "/home/laster/Programming/RustProjects/fracform-core/generated/animations/new_animation/";

    // Animation generation
    let mut factor;
    let mut fractal;
    let start = Instant::now();
    for frame in 0..frame_count {
        println!("{:.2?}%", frame as f64 / frame_count as f64 * 100.0);
        factor = f(
            frame as f64,
            0.0,
            start_factor as f64,
            frame_count as f64,
            end_factor as f64,
        );
        let formula = format!("z * c - z.powf({:.4?}) + c", factor);
        create_formula_project(&formula).expect("Failed to generate Rust code");
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
        save_bitmap(
            &color_bitmap,
            &format!("{}{}_mandelbrot_animated.png", animation_foulder, frame),
        );
    }
    println!("Animation took {:.2?}", start.elapsed());
}

fn run() {
    let start = Instant::now();
    if create_formula_project("z * c - z.powf(5.0) + c").expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();
    println!("Library loaded in {:.2?}", start.elapsed());
    let fractal = Fractal::new(
        1000,
        1000,
        0.7,
        Complex::new(0.0, 0.0),
        1000,
        160,
        Some(Complex::new(-0.9, 0.27)),
        // PaletteMode::Smooth { shift: Some(200), offset: Some(66) },
        // PaletteMode::BrownAndBlue,
        PaletteMode::Rainbow { offset: Some(205) },
        // PaletteMode::Custom,
        // PaletteMode::Naive { shift: Some(100), offset: Some(10) },
    );
    unsafe {
        JOBS = 128;
    }

    let mandelbrot = fractal.clone().mandelbrot();
    let color_bitmap = fractal.make_color_from_bitmap(mandelbrot);
    save_bitmap(&color_bitmap, "output.png");

    // let fractal = Fractal::new(
    //     1000,
    //     1000,
    //     0.5,
    //     Complex::new(0.0, 0.0),
    //     5000,
    //     40.0,
    //     Some(Complex::new(0.8, 0.44)),
    // );
    //
    // let julia = fractal.clone().julia();
    // let color_bitmap = fractal.make_color_from_bitmap(julia);
    // save_bitmap(&color_bitmap);
}

fn main() {
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(make_animation)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}
