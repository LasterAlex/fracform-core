use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::config::{JOBS, MAX_PIXELS};

use super::*;

fn z_escaped(z: Complex<f64>, max_abs: u32) -> bool {
    z.norm_sqr() as u32 > max_abs
}

impl Fractal {
    pub fn buddhabrot_worker(&self, bitmap: &Arc<Mutex<Bitmap>>, tasks: &[Complex<f64>]) {
        for c in tasks.iter() {
            let mut z = Complex::new(0.0, 0.0);

            let mut i = 0;
            while i < self.iterations {
                z = function(z, *c);
                i += 1;

                let z_pix = self.coord_to_pix(z);
                let z_pix_x = z_pix.0 as i32;
                let z_pix_y = z_pix.1 as i32;
                if z_pix_x < self.width && z_pix_y < self.height && z_pix_x >= 0 && z_pix_y >= 0 {
                    bitmap.lock().unwrap()[(z_pix_x * self.height + z_pix_y) as usize] += 1;
                } else if z_escaped(z, self.max_abs) {
                    break; // We need both cuz there are fractals that 'loop around'
                           // and return back to the screen
                }
            }
        }
    }

    pub fn buddhabrot(&self, fractal_type: FractalType) -> Bitmap {
        let buddhabrot_bitmap = Arc::new(Mutex::new([0; MAX_PIXELS as usize]));

        let start = Instant::now();
        let FractalType::Buddhabrot { rounds } = fractal_type else {
            panic!("Invalid fractal type, buddhabrot?");
        };
        let mandelbrot_bitmap = self.clone().mandelbrot();
        let mut current_x = 0.0;
        let mut current_y = 0.0;
        let mut c = self.pix_to_coord(current_x as i32, current_y as i32);
        let bottom_left_corner = self.pix_to_coord(self.width - 1, self.height - 1);
        let mut tasks = vec![];
        let x_pix_step = self.width as f64 / (rounds as f64).sqrt();
        let y_pix_step = self.height as f64 / (rounds as f64).sqrt();
        let x_coord_step = (bottom_left_corner.re - c.re) / (rounds as f64).sqrt();
        let y_coord_step = (bottom_left_corner.im - c.im) / (rounds as f64).sqrt();
        let zeroth_y_pixel = self.pix_to_coord(0, 0).im;
        loop {
            if mandelbrot_bitmap[(current_x as i32 * self.height + (current_y as i32)) as usize]
                != self.iterations
            {
                tasks.push(c);
            }
            current_y += y_pix_step;
            c.im += y_coord_step;
            if (current_x as i32) >= (self.width) && (current_y as i32) >= (self.height) {
                break;
            }
            if current_y >= self.height.into() {
                current_x += x_pix_step;
                current_y = 0.0;
                c.re += x_coord_step;
                c.im = zeroth_y_pixel;
            }
        }

        println!(
            "Found {} points in {}ms",
            tasks.len(),
            start.elapsed().as_millis()
        );

        let chunk_size;
        unsafe {
            chunk_size = tasks.len() / JOBS as usize;
        }

        let start = Instant::now();

        tasks.par_chunks(chunk_size).for_each(|chunk| {
            self.buddhabrot_worker(&buddhabrot_bitmap, chunk);
        });

        println!("Time taken to generate buddhabrot: {:.2?}", start.elapsed());

        let binding = *buddhabrot_bitmap.lock().unwrap();
        binding
    }
}
