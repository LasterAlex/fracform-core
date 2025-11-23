use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgba};
use num::Complex;

use crate::{
    colors::PaletteMode,
    compare_shadows::{compare_shadows, is_bitmap_uniform, precompute_img},
    formula::{compile_formula_project, create_formula_project, load_library},
    fractals::{Fractal, FractalType},
    make_fractal,
    random_formula::{adjust_random_param, from_func_notation, Expr},
};

pub fn get_formula_score(expr_formula: Expr, img: &Vec<Vec<bool>>) -> f64 {
    let formula = from_func_notation(expr_formula);

    let width = img.len() as i32;
    let height = img[0].len() as i32;
    let zoom = 0.5;
    let center_coordinates = Complex::new(0.0, 0.0);
    let iterations = 100;

    let palette_mode = PaletteMode::BlackAndWhite;

    // println!("{}", formula);
    let fractal_type = FractalType::Mandelbrot;
    let c = Complex::new(0.0, 0.0);
    let max_abs = 1280;

    if create_formula_project(&formula).expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();

    let mut fractal = Fractal::new(
        width,
        height,
        zoom,
        center_coordinates,
        iterations,
        max_abs,
        Some(c),
        palette_mode,
    );

    let fractal_bitmap = make_fractal(&mut fractal, fractal_type);

    if is_bitmap_uniform(&fractal_bitmap) {
        return 0.0;
    }

    compare_shadows(&fractal_bitmap, img).unwrap()
}

fn push_with_limit<T>(v: &mut Vec<T>, item: T, limit: usize) {
    v.push(item);
    if v.len() > limit {
        // keep the 30 most recent (at the end)
        let remove_count = v.len() - limit;
        v.drain(0..remove_count);
    }
}

fn get_multiple_slightly_different_formulas(
    formula: Expr,
    latest_formulas: &mut Vec<Expr>,
    highest_score: f64,
    num: u32,
) -> Vec<Expr> {
    let fine_tune = highest_score >= 0.75;
    let mut formulas = vec![];
    for _ in 0..num {
        let mut new_formula = adjust_random_param(formula.clone(), fine_tune);
        new_formula = adjust_random_param(new_formula, fine_tune);
        while latest_formulas.contains(&new_formula) {
            new_formula = adjust_random_param(new_formula, fine_tune);
        }
        push_with_limit(latest_formulas, new_formula.clone(), 3000);
        formulas.push(new_formula);
    }

    formulas
}

fn prepeare_image(filename: &str) -> (image::DynamicImage, bool) {
    let img = image::open(filename).unwrap();
    let (w, h) = img.dimensions();

    // Determine scale so the *smallest* side becomes 50
    let scale = 50.0 / w.min(h) as f32;

    // Compute new dimensions
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;

    // Resize using a high-quality filter
    let resized_img = img.resize(new_w, new_h, FilterType::CatmullRom);

    find_best_rotation(&resized_img)
}

fn find_best_rotation(img: &DynamicImage) -> (DynamicImage, bool) {
    let rotated = img.rotate90();

    if calculate_split_score(&img) < calculate_split_score(&rotated) {
        return (img.clone(), false);
    } else {
        return (rotated, true);
    }
}

fn calculate_split_score(img: &DynamicImage) -> f64 {
    let (width, height) = img.dimensions();
    let mid_y = height / 2;

    // Count black pixels in top and bottom halves
    let mut top_black = 0;
    let mut bottom_black = 0;

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let is_black = is_black_pixel(&pixel);

            if y < mid_y {
                if is_black {
                    top_black += 1;
                }
            } else {
                if is_black {
                    bottom_black += 1;
                }
            }
        }
    }

    // Return the absolute difference - lower is better (more balanced)
    ((top_black as i32) - (bottom_black as i32)).abs() as f64
}

fn is_black_pixel(pixel: &Rgba<u8>) -> bool {
    // Consider a pixel black if it's closer to black than white
    let avg = (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3;
    avg < 128
}

pub fn train(filename: &str) -> (String, bool) {
    let start_formula = Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Var("z".to_string())),
            Box::new(Expr::Var("z".to_string())),
        )),
        Box::new(Expr::Var("c".to_string())),
    );
    let (img, rotate) = prepeare_image(filename);
    let img_bitmap = precompute_img(img).unwrap();

    let mut highest_formula = (
        start_formula.clone(),
        get_formula_score(start_formula.clone(), &img_bitmap),
    );

    let mut latest_formulas = vec![];

    for _ in 0..20000 {
        let formulas = get_multiple_slightly_different_formulas(
            highest_formula.0.clone(),
            &mut latest_formulas,
            highest_formula.1,
            3,
        );

        for new_formula in formulas {
            let score = get_formula_score(new_formula.clone(), &img_bitmap);
            if score > highest_formula.1 {
                highest_formula = (new_formula, score);
                println!("new highest: {score}");
            }
        }
    }
    println!("FINAL SCORE: {}", highest_formula.1);

    (from_func_notation(highest_formula.0), rotate)
}
