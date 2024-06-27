use crate::config::{BUDDHA_ITER_RGB, DIMENSIONS};
use num::Complex;
use std::cmp::min;
use std::ops::Shl;
use std::time::Instant;

pub struct IterRGB {
    r: u32,
    g: u32,
    b: u32,
}

pub struct ColorMultiplicationRGB {
    r: f64,
    g: f64,
    b: f64,
}

#[derive(PartialEq)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    Buddhabrot,
    Antibuddhabrot,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Pixel {
    x: i32,
    y: i32,
}

pub struct Fractal {
    width: i32,
    height: i32,
    zoom: f64,
    shift: Complex<f64>,
    c: Option<Complex<f64>>,
    iterations: u32,
    max_abs: f64,
    fractal_type: FractalType,
}

fn max_of_tuple(t: (u32, u32, u32)) -> u32 {
    let (a, b, c) = t;
    a.max(b).max(c)
}

fn min_of_tuple(t: (u32, u32, u32)) -> u32 {
    let (a, b, c) = t;
    a.min(b).min(c)
}

pub fn f(x: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    -((y1 - y2) * x + (x1 * y2 - x2 * y1)) / (x2 - x1)
}

pub fn hits_to_col_sqrt(val: u32, max: u32, min: u32) -> u8 {
    //3rd root gives better results
    (((val - min) as f64 / max as f64).powf(1. / 1.7) * 255.) as u8
}

pub fn hits_to_col_lin(val: u32, max: u32) -> u8 {
    ((val as f64 / max as f64) * 255.) as u8
}

fn naive_color(iterations: u32, shift: u32, max_iterations: u32) -> (u8, u8, u8, u8) {
    if iterations == max_iterations {
        // Return black for points inside the Mandelbrot set
        return (0, 0, 0, 255);
    }

    // Normalize the iteration count to a float value between 0 and 1
    let t = iterations as f32 / shift as f32;

    // Generate a smooth gradient (example using HSL to RGB conversion)
    // Adjust the parameters to get different color schemes
    let hue = 360.0 * t;
    let saturation = 1.0;
    let lightness = if t < 0.5 { 2.0 * t } else { 2.0 * (1.0 - t) };

    hsl_to_rgb(hue, saturation, lightness)
}

// Function to convert HSL to RGB
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        300..=359 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0),
    };

    let (r, g, b) = ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0);
    (r as u8, g as u8, b as u8, 255)
}

fn blue_brown_color(iterations: u32, shift: u32, max_iterations: u32) -> (u8, u8, u8, u8) {
    if iterations == max_iterations {
        return (0, 0, 0, 255);
    }

    let t = iterations as f32 / shift as f32;
    let r = (139.0 * t) as u8; // Interpolating to brown
    let g = (69.0 * t) as u8; // Interpolating to brown
    let b = (255.0 * (1.0 - t)) as u8; // Interpolating to blue

    (r, g, b, 255)
}

fn rainbow_color(iterations: u32, shift: u32, max_iterations: u32) -> (u8, u8, u8, u8) {
    if iterations == max_iterations {
        return (0, 0, 0, 255);
    }

    let t = iterations as f32 / shift as f32;
    let hue = 360.0 * t; // Full range of hue
    let saturation = 1.0;
    let lightness = 0.5;

    hsl_to_rgb(hue, saturation, lightness)
}

impl Fractal {
    pub fn new(
        width: i32,
        height: i32,
        zoom: f64,
        shift: Complex<f64>,
        iterations: u32,
        max_abs: f64,
        fractal_type: FractalType,
        c: Option<Complex<f64>>,
    ) -> Fractal {
        Fractal {
            width,
            height,
            fractal_type,
            shift,
            zoom,
            iterations,
            max_abs,
            c,
        }
    }

    fn pix_to_coord(&self, pix: &Pixel) -> Complex<f64> {
        Complex::new(
            f(
                pix.x as f64,
                (self.width / 2) as f64,
                self.shift.re,
                (self.width / 2 + self.height / 2) as f64,
                (1.0 / self.zoom) + self.shift.re,
            ),
            f(
                pix.y as f64,
                0.0,
                (1.0 / self.zoom) + self.shift.im,
                self.height as f64,
                -(1.0 / self.zoom) + self.shift.im,
            ),
        )
    }

    fn coord_to_pix(&self, z: Complex<f64>) -> Pixel {
        Pixel {
            x: f(
                z.re,
                self.shift.re,
                (self.width / 2) as f64,
                (1.0 / self.zoom) + self.shift.re,
                (self.width / 2 + self.height / 2) as f64,
            ) as i32,
            y: f(
                z.im,
                (1.0 / self.zoom) + self.shift.im,
                0.0,
                -(1.0 / self.zoom) + self.shift.im,
                self.height as f64,
            ) as i32,
        }
    }

    fn function(&self, z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
        if z == Complex::new(0.0, 0.0) {
            return c;
        }
        z * z + c
    }

    pub fn make_fractal(&self) -> Vec<Vec<(u8, u8, u8, u8)>> {
        let mut bitmap = [(0 as u32, 0 as u32, 0 as u32); (DIMENSIONS.0 * DIMENSIONS.1) as usize];
        let t0 = Instant::now();
        for x in 0..self.width {
            for y in 0..self.height {
                let pix = Pixel { x, y };
                let c;
                let mut z;
                if self.fractal_type == FractalType::Mandelbrot
                    || self.fractal_type == FractalType::Buddhabrot
                    || self.fractal_type == FractalType::Antibuddhabrot
                {
                    c = self.pix_to_coord(&pix);
                    z = Complex::new(0.0, 0.0);
                } else if self.fractal_type == FractalType::Julia {
                    c = match self.c {
                        Some(c) => c,
                        None => Complex::new(0.0, 0.0),
                    };
                    z = self.pix_to_coord(&pix);
                } else {
                    c = Complex::new(0.0, 0.0);
                    z = Complex::new(0.0, 0.0);
                }

                let mut i = 0;
                while i < self.iterations {
                    z = self.function(z, c);
                    i += 1;
                    if z.norm_sqr() > self.max_abs {
                        bitmap[(x * self.width + y) as usize] = (i, i, i);
                        break;
                    }
                    if i > BUDDHA_ITER_RGB.0 && bitmap[(x * self.width + y) as usize].0 == 0 {
                        bitmap[(x * self.width + y) as usize].0 = self.iterations;
                    }
                    if i > BUDDHA_ITER_RGB.1 && bitmap[(x * self.width + y) as usize].1 == 0 {
                        bitmap[(x * self.width + y) as usize].1 = self.iterations;
                    }
                    if i > BUDDHA_ITER_RGB.2 && bitmap[(x * self.width + y) as usize].2 == 0 {
                        bitmap[(x * self.width + y) as usize].2 = self.iterations;
                    }
                }

                if i == self.iterations {
                    bitmap[(x * self.width + y) as usize] = (i, i, i);
                }
            }
        }
        println!("time {}", t0.elapsed().as_secs_f64());

        let mut buddha_bitmap =
            [(0 as u32, 0 as u32, 0 as u32); (DIMENSIONS.0 * DIMENSIONS.1) as usize];

        if self.fractal_type == FractalType::Buddhabrot
            || self.fractal_type == FractalType::Antibuddhabrot
        {
            let rounds = 200_000_000;
            let mut time = Instant::now().elapsed();
            let mut time_start;
            let mut pix_z: Pixel;
            let mut c;
            let mut z;
            let mut i;
            let top_left_corner = self.pix_to_coord(&Pixel { x: 0, y: 0 });
            let bottom_right_corner = self.pix_to_coord(&Pixel {
                x: self.width - 1,
                y: self.height - 1,
            });
            let rounds_sqrt = ((rounds as f64).sqrt().floor()) as u32;
            let mut c_pix;
            let mut does_escape = (true, true, true);
            let mut tmp;
            let pix_buff_template = (vec![], vec![], vec![]);
            let mut pix_buff = pix_buff_template.clone();
            println!("{:?}, {:?}", top_left_corner, bottom_right_corner);
            for x in 0..rounds_sqrt {
                println!("{}%", (x as f64) / (rounds_sqrt as f64) * 100.0);
                for y in 0..rounds_sqrt {
                    c = Complex::new(
                        f(
                            x as f64,
                            0.0,
                            top_left_corner.re,
                            rounds_sqrt as f64,
                            bottom_right_corner.re,
                        ),
                        f(
                            y as f64,
                            0.0,
                            top_left_corner.im,
                            rounds_sqrt as f64,
                            bottom_right_corner.im,
                        ),
                    );
                    c_pix = self.coord_to_pix(c);
                    z = Complex::new(0.0, 0.0);
                    tmp = bitmap[(c_pix.x * self.width + c_pix.y) as usize];
                    does_escape.0 = tmp.0 < self.iterations;
                    does_escape.1 = tmp.1 < self.iterations;
                    does_escape.2 = tmp.2 < self.iterations;
                    if !(self.fractal_type == FractalType::Antibuddhabrot)
                        && !does_escape.0
                        && !does_escape.1
                        && !does_escape.2
                    {
                        continue;
                    }
                    i = 0;
                    time_start = Instant::now();
                    while i < self.iterations {
                        z = self.function(z, c);
                        if z.norm_sqr() > self.max_abs {
                            break;
                        }

                        pix_z = self.coord_to_pix(z);
                        if pix_z.x < self.width
                            && pix_z.y < self.height
                            && pix_z.x >= 0
                            && pix_z.y >= 0
                        {
                            if (i < BUDDHA_ITER_RGB.0)
                                && (does_escape.0
                                    || self.fractal_type == FractalType::Antibuddhabrot)
                            {
                                pix_buff.0.push((pix_z.x * self.width + pix_z.y) as usize);
                            }
                            if (i < BUDDHA_ITER_RGB.1)
                                && (does_escape.1
                                    || self.fractal_type == FractalType::Antibuddhabrot)
                            {
                                pix_buff.1.push((pix_z.x * self.width + pix_z.y) as usize);
                            }
                            if (i < BUDDHA_ITER_RGB.2)
                                && (does_escape.2
                                    || self.fractal_type == FractalType::Antibuddhabrot)
                            {
                                pix_buff.2.push((pix_z.x * self.width + pix_z.y) as usize);
                            }
                        }

                        i += 1;
                    }
                    time += time_start.elapsed();

                    if (i < self.iterations) || self.fractal_type == FractalType::Antibuddhabrot {
                        for ind in pix_buff.0.iter() {
                            buddha_bitmap[*ind].0 += 1;
                        }
                        for ind in pix_buff.1.iter() {
                            buddha_bitmap[*ind].1 += 1;
                        }
                        for ind in pix_buff.2.iter() {
                            buddha_bitmap[*ind].2 += 1;
                        }
                    }
                    pix_buff.0.clear();
                    pix_buff.1.clear();
                    pix_buff.2.clear();
                }
            }
            println!("Spent {:?}", time);
        }

        let mut color_bitmap =
            vec![vec![(0, 0, 0, 255); self.height as usize]; self.width as usize];

        if self.fractal_type == FractalType::Buddhabrot
            || self.fractal_type == FractalType::Antibuddhabrot
        {
            let mut max = (0, 0, 0);
            let mut min_pix = self.iterations;
            for x in 0..self.width {
                for y in 0..self.height {
                    if buddha_bitmap[(x * self.width + y) as usize].0 > max.0 {
                        max.0 = buddha_bitmap[(x * self.width + y) as usize].0;
                    }
                    if buddha_bitmap[(x * self.width + y) as usize].1 > max.1 {
                        max.1 = buddha_bitmap[(x * self.width + y) as usize].1;
                    }
                    if buddha_bitmap[(x * self.width + y) as usize].2 > max.2 {
                        max.2 = buddha_bitmap[(x * self.width + y) as usize].2;
                    }
                    if min(
                        buddha_bitmap[(x * self.width + y) as usize].0,
                        min(
                            buddha_bitmap[(x * self.width + y) as usize].1,
                            buddha_bitmap[(x * self.width + y) as usize].2,
                        ),
                    ) < min_pix
                    {
                        min_pix = min(
                            buddha_bitmap[(x * self.width + y) as usize].0,
                            min(
                                buddha_bitmap[(x * self.width + y) as usize].1,
                                buddha_bitmap[(x * self.width + y) as usize].2,
                            ),
                        );
                    }
                }
            }
            println!("{}", min_pix);
            for x in 0..self.width {
                for y in 0..self.height {
                    let r = (hits_to_col_sqrt(
                        buddha_bitmap[(x * self.width + y) as usize].0 as u32,
                        // 10000,
                        (max.0) as u32,
                        min_pix,
                    ) as f64) as u8; //1.1
                    let g = (hits_to_col_sqrt(
                        buddha_bitmap[(x * self.width + y) as usize].1 as u32,
                        // 10000,
                        (max.1) as u32,
                        min_pix,
                    ) as f64) as u8; //0.8
                    let b = (hits_to_col_sqrt(
                        buddha_bitmap[(x * self.width + y) as usize].2 as u32,
                        // 10000,
                        (max.2) as u32,
                        min_pix,
                    ) as f64) as u8; //0.6
                    color_bitmap[x as usize][y as usize] = (r, g, b, 255);
                }
            }
        } else if self.fractal_type == FractalType::Mandelbrot
            || self.fractal_type == FractalType::Julia
        {
            for x in 0..self.width {
                for y in 0..self.height {
                    color_bitmap[x as usize][y as usize] = naive_color(
                        bitmap[(x * self.width + y) as usize].0,
                        400,
                        self.iterations,
                    );
                }
            }
        }

        color_bitmap
    }
}
