pub mod mandelbrot;
use cached::UnboundCache;
use std::{time::Instant, usize};

use cached::proc_macro::cached;

use num::Complex;

use crate::{
    colors::{set_color, PaletteMode},
    config::MAX_PIXELS,
    formula::execute_function,
};

pub type Bitmap = [u32; MAX_PIXELS as usize];

pub enum FractalType {
    Mandelbrot,
    Julia,
    Buddhabrot,
}

pub fn f(x: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    // Linear interpolation, very useful for a lot of things
    ((y1 - y2) * x + (x1 * y2 - x2 * y1)) / (x1 - x2)
}

#[cached(  // Speeds up the process A LOT
    ty = "UnboundCache<i32, f64>",
    create = "{ UnboundCache::new() }",
    convert = r#"{ x }"#
)]
fn x_to_coord(x: i32, width: i32, shift_x: f64, zoom: f64) -> f64 {
    f(
        x as f64,
        0.0,
        -(1.0 / zoom) + shift_x,
        width as f64,
        (1.0 / zoom) + shift_x,
    )
}

#[cached(
    ty = "UnboundCache<i32, f64>",
    create = "{ UnboundCache::new() }",
    convert = r#"{ y }"#
)]
fn y_to_coord(y: i32, height: i32, shift_y: f64, zoom: f64) -> f64 {
    f(
        y as f64,
        0.0,
        (1.0 / zoom) + shift_y,
        height as f64,
        -(1.0 / zoom) + shift_y,
    )
}

pub fn hits_to_col_sqrt(val: u32, max: u32, min: u32) -> u8 {
    // Buddhabrot code
    //3rd root gives better results
    (((val - min) as f64 / max as f64).powf(1. / 1.7) * 255.) as u8
}

pub fn hits_to_col_lin(val: u32, max: u32) -> u8 {
    ((val as f64 / max as f64) * 255.) as u8
}

#[derive(Clone)]
pub struct Fractal {
    width: i32,
    height: i32,
    zoom: f64,
    shift: Complex<f64>,
    c: Option<Complex<f64>>,
    iterations: u32,
    max_abs: u32,
    palette_mode: PaletteMode,
}

pub fn function(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
    execute_function(z, c)
}

impl Fractal {
    pub fn new(
        width: i32,
        height: i32,
        zoom: f64,
        shift: Complex<f64>,
        iterations: u32,
        max_abs: u32,
        c: Option<Complex<f64>>,
        palette_mode: PaletteMode,
    ) -> Fractal {
        if width * height > MAX_PIXELS.try_into().unwrap() {
            panic!("Too many pixels, increase MAX_PIXELS and STACK_SIZE in config.rs");
        }
        Fractal {
            width,
            height,
            shift,
            zoom,
            iterations,
            max_abs,
            c,
            palette_mode,
        }
    }

    fn pix_to_coord(&self, x: i32, y: i32) -> Complex<f64> {
        Complex::new(
            x_to_coord(x, self.width, self.shift.re, self.zoom),
            y_to_coord(y, self.height, self.shift.im, self.zoom),
        )
    }

    pub fn coord_to_pix(&self, z: Complex<f64>) -> (i32, i32) {
        (
            f(
                z.re,
                self.shift.re,
                (self.width / 2) as f64,
                (1.0 / self.zoom) + self.shift.re,
                (self.width / 2 + self.height / 2) as f64,
            ) as i32,
            f(
                z.im,
                (1.0 / self.zoom) + self.shift.im,
                0.0,
                -(1.0 / self.zoom) + self.shift.im,
                self.height as f64,
            ) as i32,
        )
    }

    pub fn make_color_from_bitmap(&self, bitmap: Bitmap) -> Vec<Vec<(u8, u8, u8)>> {
        let default_color = (0, 0, 0);
        let mut color_bitmap = vec![vec![default_color; self.width as usize]; self.height as usize];
        let start = Instant::now();
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let i = bitmap[(x * self.width as usize + y) as usize];
                if i < self.iterations {
                    color_bitmap[y][x] = set_color(i, self.iterations, self.palette_mode.clone());
                }
            }
        }
        let elapsed = start.elapsed();

        println!("Time taken to color it: {:.2?}", elapsed);

        color_bitmap
    }
}
