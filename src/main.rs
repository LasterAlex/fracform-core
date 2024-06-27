use std::{thread, time::Instant};

use colors::PaletteMode;
use config::{JOBS, STACK_SIZE};
use formula::{compile_formula_project, create_formula_project, load_library};
use fractals::Fractal;
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

    image_buffer
        .save(name)
        .expect("Failed to save image");
}

fn run() {
    let start = Instant::now();
    if create_formula_project("z * c - z.powf(5.9) + c").expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();
    println!("Library loaded in {:.2?}", start.elapsed());
    let fractal = Fractal::new(
        1000,
        1000,
        0.5,
        Complex::new(0.0, 0.0),
        1000,
        160,
        Some(Complex::new(-0.9, 0.27)),
        // PaletteMode::Smooth { shift: Some(200), offset: Some(66) },
        // PaletteMode::BrownAndBlue,
        PaletteMode::Rainbow { offset: Some(255) },
        // PaletteMode::Custom,
        // PaletteMode::Naive { shift: Some(100), offset: Some(10) },
    );
    unsafe {
        JOBS = 128;
    }

    let mandelbrot = fractal.clone().mandelbrot();
    let color_bitmap = fractal.make_color_from_bitmap(mandelbrot);
    save_bitmap(&color_bitmap, "output.bmp");

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
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}
