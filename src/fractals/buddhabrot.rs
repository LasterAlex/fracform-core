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
    pub fn buddhabrot_worker(&self, bitmap: &Arc<Mutex<Bitmap>>, tasks: &[Complex<f64>]) {
        let mut pix_buf = vec![];
        pix_buf.reserve(tasks.len() * (self.iterations).sqrt() as usize);
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
                    pix_buf.push((z_pix_x * self.height + z_pix_y) as usize);
                } else if z_escaped(z, self.max_abs) {
                    break; // We need both cuz there are fractals that 'loop around'
                           // and return back to the screen
                }
            }
            if pix_buf.len() > WRITE_TO_BITMAP_LEN_THRESHOLD {
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
        drop(lock);
    }

    // fn task_creation_worker(
    //     &self,
    //     worker_num: usize,
    //     chunk_size: usize,
    //     tasks: &Arc<Mutex<Vec<Complex<f64>>>>,
    //     mandelbrot_bitmap: &Bitmap,
    //     fractal_type: FractalType,
    // ) {
    //     let FractalType::Buddhabrot { rounds } = fractal_type else {
    //         panic!("Invalid fractal type, buddhabrot?");
    //     };
    //     let top_right_corner = self.pix_to_coord(0, 0);
    //     let bottom_left_corner = self.pix_to_coord(self.width - 1, self.height - 1);
    //     let x_pix_step = self.width as f64 / (rounds as f64).sqrt();
    //     let y_pix_step = self.height as f64 / (rounds as f64).sqrt();
    //     let x_coord_step = (bottom_left_corner.re - top_right_corner.re) / (rounds as f64).sqrt();
    //     let y_coord_step = (bottom_left_corner.im - top_right_corner.im) / (rounds as f64).sqrt();
    //     let zeroth_y_pixel = self.pix_to_coord(0, 0).im;
    //
    //     let current_place = (worker_num * chunk_size) as f64 / rounds as f64 * self.height as f64;
    //     let mut current_x = current_place.floor();
    //     let mut current_y = (current_place - current_x) * self.height as f64;
    //
    //     let next_place =
    //         ((worker_num + 1) * chunk_size) as f64 / rounds as f64 * self.height as f64;
    //     let next_x = min(next_place.floor() as i32, self.width - 1);
    //     let next_y = min(
    //         ((next_place - next_x as f64) * self.height as f64) as i32,
    //         self.height - 1,
    //     );
    //     println!(
    //         "Worker {} started at {},{}",
    //         worker_num, current_x, current_y
    //     );
    //     let mut c = self.pix_to_coord(current_x as i32, current_y as i32);
    //     let mut task_buf = vec![];
    //     task_buf.reserve(chunk_size);
    //     loop {
    //         if mandelbrot_bitmap[(current_x as i32 * self.height + (current_y as i32)) as usize]
    //             != self.iterations
    //         {
    //             task_buf.push(c);
    //         }
    //         current_y += y_pix_step;
    //         c.im += y_coord_step;
    //         if (current_x as i32) >= (next_x) && (current_y as i32) >= (next_y) {
    //             break;
    //         }
    //         if current_y >= self.height.into() {
    //             current_x += x_pix_step;
    //             current_y = 0.0;
    //             c.re += x_coord_step;
    //             c.im = zeroth_y_pixel;
    //         }
    //     }
    //
    //     tasks.lock().unwrap().extend(task_buf);
    // }

    pub fn buddhabrot(&self, fractal_type: FractalType) -> Bitmap {
        let FractalType::Buddhabrot { rounds } = fractal_type else {
            panic!("Invalid fractal type, buddhabrot?");
        };

        let mandelbrot_bitmap = self.clone().mandelbrot();
        // unsafe {
        //     let chunk_size = rounds as usize / JOBS as usize;
        //     let tasks = Arc::new(Mutex::new(vec![]));
        //     (0..JOBS).into_par_iter().for_each(|worker_num| {
        //         self.task_creation_worker(
        //             worker_num as usize,
        //             chunk_size,
        //             &tasks,
        //             &mandelbrot_bitmap,
        //             fractal_type.clone(),
        //         );
        //     });
        //     println!(
        //         "Time taken to create {:?} tasks: {:.2?}",
        //         tasks.lock().unwrap().len(),
        //         start.elapsed()
        //     );
        //     // loop {
        //     //     self.task_creation_worker(worker_num, chunk_size, &tasks, &mandelbrot_bitmap, fractal_type.clone());
        //     //     worker_num += 1;
        //     //     if worker_num == JOBS.try_into().unwrap() {
        //     //         break;
        //     //     }
        //     // }
        // }
        let start = Instant::now();
        let mut current_x = 0.0;
        let mut current_y = 0.0;
        let mut c = self.pix_to_coord(current_x as i32, current_y as i32);
        let bottom_left_corner = self.pix_to_coord(self.width - 1, self.height - 1);
        let x_pix_step = self.width as f64 / (rounds as f64).sqrt();
        let y_pix_step = self.height as f64 / (rounds as f64).sqrt();
        let x_coord_step = (bottom_left_corner.re - c.re) / (rounds as f64).sqrt();
        let y_coord_step = (bottom_left_corner.im - c.im) / (rounds as f64).sqrt();
        let zeroth_y_pixel = self.pix_to_coord(0, 0).im;

        // let tasks = vec![None::<Complex<f64>>; rounds as usize];
        // println!("Found {} points in {}ms", tasks.len(), start.elapsed().as_millis());
        // let start = Instant::now();
        let mut tasks = vec![];
        tasks.reserve(rounds as usize);
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
        let buddhabrot_bitmap = Arc::new(Mutex::new([0; MAX_PIXELS as usize]));

        println!(
            "Found {} points in {}ms",
            tasks.len(),
            start.elapsed().as_millis()
        );
        // self.buddhabrot_worker(&mut buddhabrot_bitmap, &tasks);
        // println!("Time taken to generate buddhabrot: {:.2?}", start.elapsed());
        // return buddhabrot_bitmap;

        // I could NOT get the parallelization to work effectively at ALL, it was only making stuff
        // slower, Mutex really slows everything down, cuz a LOT of everything needs to be copied
        let start = Instant::now();

        let chunk_size;
        unsafe {
            if JOBS == 1 {
                self.buddhabrot_worker(&buddhabrot_bitmap, &tasks);
                println!("Time taken to generate buddhabrot: {:.2?}", start.elapsed());
                return *buddhabrot_bitmap.lock().unwrap();
            }

            chunk_size = tasks.len() / JOBS as usize;
        }

        tasks.par_chunks(chunk_size).for_each(|chunk| {
            self.buddhabrot_worker(&buddhabrot_bitmap, chunk);
        });

        println!("Time taken to generate buddhabrot: {:.2?}", start.elapsed());

        return *buddhabrot_bitmap.lock().unwrap();
    }
}
