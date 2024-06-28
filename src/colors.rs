use cached::UnboundCache;
use std::cmp::min;

use cached::proc_macro::cached;

use crate::fractals::f;

#[derive(Clone)]
pub enum PaletteMode {
    BrownAndBlue,
    Rainbow {
        offset: Option<u32>,
    },
    Smooth {
        shift: Option<u32>,
        offset: Option<u32>,
    },
    Naive {
        shift: Option<u32>,
        offset: Option<u32>,
    },
    Custom,
}

pub fn naive_color(iterations: u32, max_iterations: u32, shift: u32, offset: u32) -> (u8, u8, u8) {
    let t = ((iterations + offset) as f64) / shift as f64;

    // Generate a smooth gradient
    let hue = 360.0 * t;
    let saturation = 1.0;
    let lightness = f(
        iterations as f64 / max_iterations as f64,
        0.0,
        0.3,
        1.0,
        0.2,
    );
    let hsl_color = ColorHSV::new(hue, saturation, lightness);

    hsl_to_rgb(hsl_color)
}

fn hsl_to_rgb(hsl_color: ColorHSV) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * hsl_color.lightness - 1.0).abs()) * hsl_color.saturation;
    let x = c * (1.0 - ((hsl_color.hue / 60.0) % 2.0 - 1.0).abs());
    let m = hsl_color.lightness - c / 2.0;

    let (r, g, b) = match hsl_color.hue as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        300..=359 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0),
    };

    let (r, g, b) = ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0);
    (r as u8, g as u8, b as u8)
}

#[derive(Clone)]
struct ColorHSV {
    hue: f64,
    saturation: f64,
    lightness: f64,
}

impl ColorHSV {
    fn new(hue: f64, saturation: f64, lightness: f64) -> Self {
        Self {
            hue,
            saturation,
            lightness,
        }
    }
}

fn create_palette(lst_of_colors_hsv: Vec<ColorHSV>, length: usize) -> Vec<(u8, u8, u8)> {
    // Needs some updating, legacy code
    let mut y = Vec::new();
    let len_segment = length / lst_of_colors_hsv.len();
    let mut lst = lst_of_colors_hsv.clone();
    lst.pop();

    for (i, color) in lst.iter().enumerate() {
        for j in 0..len_segment {
            let next_color = &lst_of_colors_hsv[i + 1];
            let g = ColorHSV::new(
                f(j as f64, 0.0, color.hue, len_segment as f64, next_color.hue),
                f(
                    j as f64,
                    0.0,
                    color.saturation,
                    len_segment as f64,
                    next_color.saturation,
                ),
                f(
                    j as f64,
                    0.0,
                    color.lightness,
                    len_segment as f64,
                    next_color.lightness,
                ),
            );
            y.push(hsl_to_rgb(g));
        }
    }

    y
}

#[cached]
fn create_custom_pallete() -> Vec<(u8, u8, u8)> {
    // Needs some updating, legacy code
    create_palette(
        vec![
            ColorHSV::new(342.0, 92.0, 71.0),
            ColorHSV::new(342.0, 0.0, 100.0),
            ColorHSV::new(205.0, 63.0, 95.0),
            ColorHSV::new(92.0, 61.0, 90.0),
            ColorHSV::new(53.0, 80.0, 100.0),
            ColorHSV::new(342.0 - 360.0, 92.0, 71.0),
        ],
        50,
    )
}

fn iters_to_color(iters: u32, max_iterations: u32, offset: u32) -> (u8, u8, u8) {
    let iters = max_iterations - iters;
    // It came to me in a dream
    let value = (iters << 21) + (iters << 10) + iters * 8 + offset;
    let red = ((value >> 16) & 0xFF) as u8;
    let green = ((value >> 8) & 0xFF) as u8;
    let blue = (value & 0xFF) as u8;

    (red, green, blue)
}

#[cached(
    ty = "UnboundCache<u32, (u8, u8, u8)>",
    create = "{ UnboundCache::new() }",
    convert = r#"{ iters }"#
)]
pub fn set_color(iters: u32, max_iterations: u32, palette_mode: PaletteMode) -> (u8, u8, u8) {
    match palette_mode {
        PaletteMode::BrownAndBlue => {
            let colors = [
                (66, 30, 15),
                (25, 7, 26),
                (9, 1, 47),
                (4, 4, 73),
                (0, 7, 100),
                (12, 44, 138),
                (24, 82, 177),
                (57, 125, 209),
                (134, 181, 229),
                (211, 236, 248),
                (241, 233, 191),
                (248, 201, 95),
                (255, 170, 0),
                (204, 128, 0),
                (153, 87, 0),
                (106, 52, 3),
            ];
            return colors[iters as usize % colors.len()];
        }
        PaletteMode::Rainbow { offset } => {
            iters_to_color(iters, max_iterations, offset.unwrap_or(0))
        }
        PaletteMode::Smooth { shift, offset } => {
            let n = (iters + offset.unwrap_or(0)) as f64 / shift.unwrap_or(max_iterations) as f64;
            let t = 1.0 - (n - n.trunc());
            let r = min(255, (9.0 * (1.0 - t) * t.powi(3) * 255.0).round() as u8);
            let g = min(
                255,
                (15.0 * (1.0 - t).powi(2) * t.powi(2) * 255.0).round() as u8,
            );
            let b = min(255, (8.5 * (1.0 - t).powi(3) * t * 255.0).round() as u8);
            (r, g, b)
        }
        PaletteMode::Naive { shift, offset } => naive_color(
            iters,
            max_iterations,
            shift.unwrap_or(max_iterations),
            offset.unwrap_or(0),
        ),
        PaletteMode::Custom => {
            let custom_palette = create_custom_pallete();
            return custom_palette[iters as usize % custom_palette.len()];
        }
    }
}
