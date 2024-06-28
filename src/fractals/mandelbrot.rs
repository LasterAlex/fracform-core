use rayon::prelude::*;
use std::time::Instant;

use crate::config::{JOBS, MAX_PIXELS};

use super::*;

fn z_escaped(z: Complex<f64>, max_abs: u32) -> bool {
    z.norm_sqr() as u32 > max_abs
}

impl Fractal {
    pub fn mandelbrot_worker(&self, chunk: &mut [u32], tasks: Vec<(i32, i32)>) {
        for (index, (x, y)) in tasks.iter().enumerate() {
            let c = self.pix_to_coord(*x, *y);
            let mut z = Complex::new(0.0, 0.0);
            let mut i = 0;
            while (i < self.iterations) && !z_escaped(z, self.max_abs) {
                z = function(z, c);
                i += 1;
            }
            chunk[index] = i;
        }
    }

    pub fn julia_worker(&self, chunk: &mut [u32], tasks: Vec<(i32, i32)>) {
        for (index, (x, y)) in tasks.iter().enumerate() {
            let mut z = self.pix_to_coord(*x, *y);
            let c = self.c.unwrap_or(Complex::new(0.0, 0.0));
            let mut i = 0;
            while (i < self.iterations) && !z_escaped(z, self.max_abs) {
                z = function(z, c);
                i += 1;
            }
            chunk[index] = i;
        }
    }

    pub fn mandelbrot_or_julia(self, fractal_type: FractalType) -> Bitmap {
        let mut bitmap = [0; MAX_PIXELS as usize];
        let mut pixels = vec![];
        for y in 0..self.height {
            for x in 0..self.width {
                pixels.push((x, y));
            }
        }
        let start = Instant::now();
        let chunk_size;
        unsafe {
            // Cuz it's working with mut static JOBS, it's ok
            if JOBS == 1 {
                match fractal_type {
                    FractalType::Mandelbrot => self.mandelbrot_worker(&mut bitmap, pixels),
                    FractalType::Julia => self.julia_worker(&mut bitmap, pixels),
                    _ => panic!("Invalid fractal type, mandelbrot or julia?"),
                }
                println!("Time taken to generate: {:.2?}", start.elapsed());
                return bitmap;
            }
            chunk_size = (self.width * self.height / JOBS as i32) as usize;
        }
        bitmap
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(index, chunk)| {
                let current_bitmap_index = (index * chunk_size) as i32;
                let next_bitmap_index = ((index + 1) * chunk_size) as i32;
                let tasks = pixels
                    .iter()
                    .skip(current_bitmap_index as usize)
                    .take(next_bitmap_index as usize - current_bitmap_index as usize)
                    .map(|(x, y)| (*x, *y))
                    .collect::<Vec<(i32, i32)>>();
                if tasks.is_empty() {
                    return;
                }

                match fractal_type {
                    FractalType::Mandelbrot => self.mandelbrot_worker(chunk, tasks),
                    FractalType::Julia => self.julia_worker(chunk, tasks),
                    _ => panic!("Invalid fractal type, mandelbrot or julia?"),
                }
            });

        println!("Time taken to generate: {:.2?}", start.elapsed());

        bitmap
    }

    pub fn mandelbrot(self) -> Bitmap {
        self.mandelbrot_or_julia(FractalType::Mandelbrot)
    }

    pub fn julia(self) -> Bitmap {
        self.mandelbrot_or_julia(FractalType::Julia)
    }
}
