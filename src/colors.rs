use cached::UnboundCache;
use std::cmp::min;

use cached::proc_macro::cached;

use crate::fractals::f;

#[derive(Clone, PartialEq)]
pub enum PaletteMode {
    BrownAndBlue,
    BlackAndWhite,
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
    GrayScale {
        shift: Option<u32>,
        uniform_factor: Option<f64>,
    },
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

#[inline]
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

pub fn generate_palette(key_colors: Vec<(u8, u8, u8)>, palette_len: usize) -> Vec<(u8, u8, u8)> {
    let n = key_colors.len();
    assert!(n >= 2, "At least two key colors required");

    let mut palette = Vec::with_capacity(palette_len);

    for i in 0..palette_len {
        // Normalized position along the palette [0..1)
        let pos = i as f32 / palette_len as f32;

        // Map pos into the segment between two key colors
        let segment = pos * n as f32;
        let idx = segment.floor() as usize % n;
        let t = segment - segment.floor();

        let (r1, g1, b1) = key_colors[idx];
        let (r2, g2, b2) = key_colors[(idx + 1) % n];

        palette.push((lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t)));
    }

    palette
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
    ty = "UnboundCache<(u32, u32), (u8, u8, u8)>",
    create = "{ UnboundCache::new() }",
    convert = r#"{ (param, max_param) }"#
)]
pub fn set_color(param: u32, max_param: u32, palette_mode: PaletteMode) -> (u8, u8, u8) {
    match palette_mode {
        PaletteMode::BrownAndBlue => {
            if param >= max_param {
                return (0, 0, 0);
            }
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
            colors[param as usize % colors.len()]
        }
        PaletteMode::BlackAndWhite => {
            if param >= max_param {
                return (0, 0, 0);
            }
            return (255, 255, 255);
        }
        PaletteMode::Rainbow { offset } => {
            if param >= max_param {
                return (0, 0, 0);
            }
            iters_to_color(param, max_param, offset.unwrap_or(0))
        }
        PaletteMode::Smooth { shift, offset } => {
            if param >= max_param {
                return (0, 0, 0);
            }
            let n = (param + offset.unwrap_or(0)) as f64 / shift.unwrap_or(max_param) as f64;
            let t = 1.0 - (n - n.trunc());
            let r = min(255, (9.0 * (1.0 - t) * t.powi(3) * 255.0).round() as u8);
            let g = min(
                255,
                (15.0 * (1.0 - t).powi(2) * t.powi(2) * 255.0).round() as u8,
            );
            let b = min(255, (8.5 * (1.0 - t).powi(3) * t * 255.0).round() as u8);
            (r, g, b)
        }
        PaletteMode::Naive { shift, offset } => {
            if param >= max_param {
                return (0, 0, 0);
            }
            naive_color(
                param,
                max_param,
                shift.unwrap_or(max_param),
                offset.unwrap_or(0),
            )
        }
        PaletteMode::Custom => {
            if param >= max_param {
                return (0, 0, 0);
            }
            let custom_palette = generate_palette(
                vec![
                    (34, 67, 76),
                    (115, 106, 76),
                    (155, 117, 76),
                    (141, 103, 76),
                    (134, 61, 76),
                    (197, 181, 76),
                ],
                100,
            );
            custom_palette[param as usize % custom_palette.len()]
        }
        PaletteMode::GrayScale {
            shift,
            uniform_factor,
        } => {
            let gray = min(
                255,
                ((param as f64 / shift.unwrap_or(max_param) as f64)
                    .powf(uniform_factor.unwrap_or(1.0))
                    * 255.0)
                    .round() as u8,
            );
            (gray, gray, gray)
        }
    }
}
