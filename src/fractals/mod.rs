pub mod buddhabrot;
pub mod mandelbrot;
pub mod nebulabrot;
use cached::{Cached, UnboundCache};
use std::{mem::discriminant, time::Instant};

use cached::proc_macro::cached;

use num::Complex;

use crate::{
    colors::{self, set_color, PaletteMode},
    config::MAX_PIXELS,
    formula::execute_function,
};

pub type Bitmap = [u32; MAX_PIXELS as usize];

#[derive(Clone, PartialEq)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    Buddhabrot {
        rounds: u32,
    },
    Antibuddhabrot {
        rounds: u32,
    },
    Nebulabrot {
        rounds: u32,
        red_iters: u32,
        green_iters: u32,
        blue_iters: u32,
        color_shift: Option<u32>,
        uniform_factor: Option<f64>,
    },
    Antinebulabrot {
        rounds: u32,
        red_iters: u32,
        green_iters: u32,
        blue_iters: u32,
        color_shift: Option<u32>,
        uniform_factor: Option<f64>,
    },
}

pub fn f(x: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    // Linear interpolation, very useful for a lot of things
    ((y1 - y2) * x + (x1 * y2 - x2 * y1)) / (x1 - x2)
}

#[cached(  // Speeds up the process A LOT
    ty = "UnboundCache<(i32, i32, i32), f64>",
    create = "{ UnboundCache::new() }",
    convert = r#"{ (x, width, height) }"#
)]
pub fn x_to_coord(x: i32, width: i32, height: i32, shift_x: f64, zoom: f64) -> f64 {
    let end_goal; // We need the center of the fractal to not be distorted, so we go with the
                  // minimum of the width and height, to preserve all of the fractal in the image
    let offset; // To center the image
    if width > height {
        offset = (width - height) as f64 / 2.0;
        end_goal = height as f64;
    } else {
        offset = 0.0;
        end_goal = width as f64;
    }
    f(
        x as f64 - offset,
        0.0,
        -(1.0 / zoom) + shift_x,
        end_goal,
        (1.0 / zoom) + shift_x,
    )
}

#[cached(
    ty = "UnboundCache<(i32, i32, i32), f64>",
    create = "{ UnboundCache::new() }",
    convert = r#"{ (y, width, height) }"#
)]
pub fn y_to_coord(y: i32, width: i32, height: i32, shift_y: f64, zoom: f64) -> f64 {
    let offset;
    let end_goal;
    if height > width {
        offset = (height - width) as f64 / 2.0;
        end_goal = width as f64;
    } else {
        offset = 0.0;
        end_goal = height as f64;
    }
    f(
        y as f64 - offset,
        0.0,
        (1.0 / zoom) + shift_y,
        end_goal,
        -(1.0 / zoom) + shift_y,
    )
}

pub fn x_coord_to_pix(x: f64, width: i32, height: i32, shift_x: f64, zoom: f64) -> f64 {
    let end_goal;
    let offset;
    if width > height {
        offset = (width - height) as f64 / 2.0;
        end_goal = height as f64;
    } else {
        offset = 0.0;
        end_goal = width as f64;
    }
    f(
        x,
        -(1.0 / zoom) + shift_x,
        0.0,
        (1.0 / zoom) + shift_x,
        end_goal,
    ) + offset
}

pub fn y_coord_to_pix(y: f64, width: i32, height: i32, shift_y: f64, zoom: f64) -> f64 {
    let offset;
    let end_goal;
    if height > width {
        offset = (height - width) as f64 / 2.0;
        end_goal = width as f64;
    } else {
        offset = 0.0;
        end_goal = height as f64;
    }
    f(
        y,
        (1.0 / zoom) + shift_y,
        0.0,
        -(1.0 / zoom) + shift_y,
        end_goal,
    ) + offset
}

fn sort<A, T>(mut array: A) -> A
where
    A: AsMut<[T]>,
    T: Ord,
{
    let slice = array.as_mut();
    slice.sort();

    array
}

#[derive(Clone)]
pub struct Fractal {
    pub width: i32,
    pub height: i32,
    pub zoom: f64,
    pub shift: Complex<f64>,
    pub c: Option<Complex<f64>>,
    pub iterations: u32,
    pub max_abs: u32,
    pub palette_mode: PaletteMode,
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
        let max_pixels: usize = MAX_PIXELS.try_into().unwrap();
        if (width as usize) * (height as usize) > max_pixels {
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
            x_to_coord(x, self.width, self.height, self.shift.re, self.zoom),
            y_to_coord(y, self.width, self.height, self.shift.im, self.zoom),
        )
    }

    pub fn coord_to_pix(&self, z: Complex<f64>) -> (f64, f64) {
        (
            x_coord_to_pix(z.re, self.width, self.height, self.shift.re, self.zoom),
            y_coord_to_pix(z.im, self.width, self.height, self.shift.im, self.zoom),
        )
    }

    pub fn make_color_from_bitmap(&self, bitmap: Bitmap) -> Vec<Vec<(u8, u8, u8)>> {
        let default_color = (0, 0, 0);
        let mut color_bitmap = vec![vec![default_color; self.height as usize]; self.width as usize];
        let mut max_param = self.iterations;
        if discriminant(&self.palette_mode)
            == discriminant(&PaletteMode::GrayScale {
                shift: None,
                uniform_factor: None,
            })
        {
            let mut tmp: Vec<&u32> = bitmap
                .iter()
                .take(self.width as usize * self.height as usize)
                .collect();
            sort(&mut tmp);
            max_param = *tmp[tmp.len() - 100];
        }
        // let start = Instant::now();
        for x in 0..self.width as usize {
            for y in 0..self.height as usize {
                let i = bitmap[x * self.height as usize + y];
                color_bitmap[x][y] = set_color(i, max_param, self.palette_mode.clone());
            }
        }
        // let elapsed = start.elapsed();

        // println!("Time taken to color it: {elapsed:.2?}");

        color_bitmap
    }
}

pub fn reset_cache() {
    Y_TO_COORD.lock().unwrap().cache_reset();
    X_TO_COORD.lock().unwrap().cache_reset();
    colors::SET_COLOR.lock().unwrap().cache_reset();
}
