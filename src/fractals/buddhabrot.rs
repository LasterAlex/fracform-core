use num::integer::Roots;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::config::{JOBS, MAX_PIXELS, WRITE_TO_BITMAP_LEN_THRESHOLD};

use super::*;

fn z_escaped(z: Complex<f64>, max_abs: u32) -> bool {
    z.norm_sqr() as u32 > max_abs
}

impl Fractal {
    pub fn buddhabrot_worker(
        &self,
        bitmap: &Arc<Mutex<Bitmap>>,
        tasks: &[Complex<f64>],
        is_antibuddhabrot: bool,
    ) {
        let mut pix_buf = vec![];
        let mut add_buf = vec![];
        pix_buf.reserve(tasks.len() * (self.iterations).sqrt() as usize);
        for c in tasks.iter() {
            add_buf.clear();
            add_buf.reserve(self.iterations as usize);
            let mut z = Complex::new(0.0, 0.0);

            let mut i = 0;
            while i < self.iterations {
                z = function(z, *c);
                i += 1;

                let z_pix = self.coord_to_pix(z);
                let z_pix_x = z_pix.0 as i32;
                let z_pix_y = z_pix.1 as i32;
                if i > 1  // This is to avoid the grid lines, try removing it and zoom in where
                // there is not much color
                    && z_pix_x < self.width
                    && z_pix_y < self.height
                    && z_pix_x >= 0
                    && z_pix_y >= 0
                {
                    add_buf.push((z_pix_x * self.height + z_pix_y) as usize);
                } else if z_escaped(z, self.max_abs) {
                    break; // We need both cuz there are fractals that 'loop around'
                           // and return back to the screen
                }
            }
            if (i == self.iterations && is_antibuddhabrot)
                || (!is_antibuddhabrot && i < self.iterations)
            // If it didn't escape we dont need it
            {
                pix_buf.extend_from_slice(&add_buf);
            }
            if pix_buf.len() > WRITE_TO_BITMAP_LEN_THRESHOLD {
                // Otherwise it will fill up to the
                // limit of RAM and swap
                let mut lock = bitmap.lock().unwrap();
                for pix in pix_buf.iter() {
                    lock[*pix] += 1;
                }
                pix_buf.clear();
                drop(lock);
            }
        }
        let mut lock = bitmap.lock().unwrap();
        for pix in pix_buf.iter() {
            lock[*pix] += 1;
        }
    }

    fn task_creation_worker(
        &self,
        worker_num: usize,
        mandelbrot_bitmap: &Bitmap,
        fractal_type: FractalType,
    ) -> Vec<Complex<f64>> {
        let is_antibuddhabrot =
            discriminant(&fractal_type) == discriminant(&FractalType::Antibuddhabrot { rounds: 0 });
        let rounds = match fractal_type {
            FractalType::Buddhabrot { rounds } => rounds,
            FractalType::Antibuddhabrot { rounds } => rounds,
            _ => {
                panic!("Invalid fractal type, buddhabrot or antibuddhabrot?")
            }
        };
        // First, we find the corners, to get the equivalent of self.width and self.height
        let top_right_corner = self.pix_to_coord(0, 0);
        let bottom_left_corner = self.pix_to_coord(self.width, self.height);
        let coord_width = bottom_left_corner.re - top_right_corner.re;
        let coord_height = top_right_corner.im - bottom_left_corner.im;

        let jobs_lock = *JOBS.lock().unwrap();

        // Then we find the coordinates of the current worker by simple interpolation
        let current_x_coord =
            top_right_corner.re + worker_num as f64 / jobs_lock as f64 * coord_width;
        let current_y_coord =
            top_right_corner.im - worker_num as f64 / jobs_lock as f64 * coord_height;

        // Calculating the pixel by c is too slow, we will just add the pix step the same way
        let x_pix_step = self.width as f64 / (rounds as f64).sqrt();
        let y_pix_step = self.height as f64 / (rounds as f64).sqrt();
        let x_coord_step = (bottom_left_corner.re - top_right_corner.re) / (rounds as f64).sqrt();
        let y_coord_step = (bottom_left_corner.im - top_right_corner.im) / (rounds as f64).sqrt();
        let zeroth_y_coord = self.pix_to_coord(0, 0).im;

        // To find, when to stop the worker
        let next_x_coord = current_x_coord + coord_width / jobs_lock as f64;
        let next_y_coord = current_y_coord - coord_height / jobs_lock as f64;

        let mut c = Complex::new(current_x_coord, current_y_coord);
        let mut current_pixel = self.coord_to_pix(c);
        // println!(
        //     "Worker {} starting at {}, {},  next is {}, {}, in {:.2?}",
        //     worker_num,
        //     c.re,
        //     c.im,
        //     next_x_coord,
        //     next_y_coord,
        //     start.elapsed()
        // );
        let mut task_buf = vec![];
        loop {
            let mandelbrot_pixel = mandelbrot_bitmap  // The said optimization
                [(current_pixel.0 as i32 * self.height + (current_pixel.1 as i32)) as usize];

            if (!is_antibuddhabrot && mandelbrot_pixel < self.iterations)
                || (is_antibuddhabrot && mandelbrot_pixel >= self.iterations)
            {
                task_buf.push(c);
            }
            c.im += y_coord_step;
            current_pixel.1 += y_pix_step;
            if (c.re) >= (next_x_coord) && (c.im) <= (next_y_coord) {
                // The height goes down, the width goes right, that's why the signs are different
                break;
            }
            if c.im <= bottom_left_corner.im {
                current_pixel.0 += x_pix_step;
                current_pixel.1 = 0.0;
                c.re += x_coord_step;
                c.im = zeroth_y_coord;
            }
        }

        task_buf
    }

    fn generate_buddhabrot_tasks(&self, fractal_type: FractalType) -> Vec<Vec<Complex<f64>>> {
        // For optimization, we first generate a mandelbrot set, and only then generate the
        // buddhabrot, excluding the points that are in the set (or including them if we want
        // antibuddhabrot).
        let mandelbrot_bitmap = self.clone().mandelbrot();
        let start = Instant::now();

        let mut tmp: Vec<Vec<Complex<f64>>> = vec![];
        (0..*JOBS.lock().unwrap())
            .into_par_iter()
            .map(|worker_num| {
                self.task_creation_worker(
                    worker_num as usize,
                    &mandelbrot_bitmap,
                    fractal_type.clone(),
                )
            })
            .collect_into_vec(&mut tmp);
        println!(
            "Time taken to create {:?} tasks: {:.2?}",
            tmp.clone().into_iter().flatten().count(),
            start.elapsed()
        );
        tmp
    }

    pub fn buddhabrot_or_antibuddhabrot(&self, fractal_type: FractalType) -> Bitmap {
        reset_cache();
        let tasks = self.generate_buddhabrot_tasks(fractal_type.clone());
        let is_antibuddhabrot =
            discriminant(&fractal_type) == discriminant(&FractalType::Antibuddhabrot { rounds: 0 });

        let start = Instant::now();

        let buddhabrot_bitmap = Arc::new(Mutex::new([0; MAX_PIXELS as usize]));
        if tasks.len() == 1 {
            self.buddhabrot_worker(&buddhabrot_bitmap, &tasks[0], is_antibuddhabrot);
            println!("Time taken to generate buddhabrot: {:.2?}", start.elapsed());
            return *buddhabrot_bitmap.lock().unwrap();
        }

        tasks.par_iter().for_each(|chunk| {
            self.buddhabrot_worker(&buddhabrot_bitmap, chunk, is_antibuddhabrot);
        });

        println!("Time taken to generate buddhabrot: {:.2?}", start.elapsed());

        return *buddhabrot_bitmap.lock().unwrap();
    }

    pub fn buddhabrot(&self, rounds: u32) -> Bitmap {
        self.buddhabrot_or_antibuddhabrot(FractalType::Buddhabrot { rounds })
    }

    pub fn antibuddhabrot(&self, rounds: u32) -> Bitmap {
        self.buddhabrot_or_antibuddhabrot(FractalType::Antibuddhabrot { rounds })
    }
}
