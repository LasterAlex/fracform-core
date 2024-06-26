mod config;
mod fractal;

use crate::config::{DIMENSIONS, ITERS, STACK_SIZE};
use crate::fractal::{Fractal, FractalType};
use image::{ImageBuffer, Rgba};
use num::Complex;
use std::thread;

fn save_bitmap(bitmap: &Vec<Vec<(u8, u8, u8, u8)>>) {
    let mut image_buffer =
        ImageBuffer::<Rgba<u8>, Vec<u8>>::new(bitmap.len() as u32, bitmap[0].len() as u32);

    for (x, row) in bitmap.iter().enumerate() {
        for (y, pixel) in row.iter().enumerate() {
            let (r, g, b, a) = *pixel;
            image_buffer.put_pixel(x as u32, y as u32, Rgba([r, g, b, a]));
        }
    }

    image_buffer
        .save("output.bmp")
        .expect("Failed to save image");
}

fn run() {
    let fractal = Fractal::new(
        DIMENSIONS.0,
        DIMENSIONS.1,
        0.7,
        Complex::new(0.4, -0.7),
        ITERS,
        160.0,
        FractalType::Julia,
        Some(Complex::new(-0.17, -0.4954)),
    );
    let start = std::time::Instant::now();
    let bitmap = fractal.make_fractal();
    println!("{:?}", std::time::Instant::now().duration_since(start));
    save_bitmap(&bitmap);
}

fn main() {
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}
