use std::{mem::discriminant, thread};

use num::Complex;
use rand::{seq::IndexedRandom, Rng};
use rand_distr::{Distribution, Normal};

use crate::{
    compare_shadows::is_bitmap_uniform, config::STACK_SIZE, formula::{compile_formula_project, create_formula_project, load_library}, fractals::Fractal, make_fractal
};

pub fn from_func_notation(expr: Expr) -> String {
    fn f(e: &Expr) -> String {
        match e {
            Expr::Num(n) => {
                let s = ((n * 1000.0).round() / 1000.0).to_string();
                // Remove trailing .0 for integers
                if s.ends_with(".0") {
                    s.trim_end_matches(".0").to_string()
                } else {
                    s
                }
            }
            Expr::Var(v) => v.clone(),

            Expr::Add(a, b) => format!("{} + {}", f(a), f(b)),
            Expr::Mul(a, b) => format!(
                "{} * {}",
                match &**a {
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Div(_, _) => format!("({})", f(a)),
                    _ => f(a),
                },
                match &**b {
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Div(_, _) => format!("({})", f(b)),
                    _ => f(b),
                }
            ),
            Expr::Sub(a, b) => format!(
                "{} - {}",
                f(a),
                match &**b {
                    Expr::Add(_, _) | Expr::Sub(_, _) => format!("({})", f(b)),
                    _ => f(b),
                }
            ),
            Expr::Div(a, b) => format!(
                "{} / {}",
                match &**a {
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Div(_, _) => format!("({})", f(a)),
                    _ => f(a),
                },
                match &**b {
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Div(_, _) | Expr::Mul(_, _) =>
                        format!("({})", f(b)),
                    _ => f(b),
                }
            ),
            Expr::Pow(a, b) => format!("pow({}, {})", f(a), f(b)),

            Expr::Imag(x) => format!("imag({})", f(x)),
            Expr::Real(x) => format!("real({})", f(x)),
            Expr::Abs(x) => format!("abs({})", f(x)),

            Expr::Exp(x) => format!("exp({})", f(x)),
            Expr::Log(x) => format!("log({})", f(x)),
            Expr::Log10(x) => format!("log10({})", f(x)),
            Expr::Sqrt(x) => format!("sqrt({})", f(x)),

            Expr::Cos(x) => format!("cos({})", f(x)),
            Expr::Sin(x) => format!("sin({})", f(x)),
            Expr::Tan(x) => format!("tan({})", f(x)),
            Expr::Acos(x) => format!("acos({})", f(x)),
            Expr::Asin(x) => format!("asin({})", f(x)),
            Expr::Atan(x) => format!("atan({})", f(x)),

            Expr::Cosh(x) => format!("cosh({})", f(x)),
            Expr::Sinh(x) => format!("sinh({})", f(x)),
            Expr::Tanh(x) => format!("tanh({})", f(x)),
            Expr::Acosh(x) => format!("acosh({})", f(x)),
            Expr::Asinh(x) => format!("asinh({})", f(x)),
            Expr::Atanh(x) => format!("atanh({})", f(x)),
        }
    }

    f(&expr)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),

    // Basics
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),

    // Complex
    Imag(Box<Expr>),
    Real(Box<Expr>),
    Abs(Box<Expr>),

    // Elementary
    Exp(Box<Expr>),
    Log(Box<Expr>),
    Log10(Box<Expr>),
    Sqrt(Box<Expr>),

    // Trig
    Cos(Box<Expr>),
    Sin(Box<Expr>),
    Tan(Box<Expr>),
    Acos(Box<Expr>),
    Asin(Box<Expr>),
    Atan(Box<Expr>),

    // Hyperbolic
    Cosh(Box<Expr>),
    Sinh(Box<Expr>),
    Tanh(Box<Expr>),
    Acosh(Box<Expr>),
    Asinh(Box<Expr>),
    Atanh(Box<Expr>),
}

pub fn gaussian_in_range(x: f64) -> f64 {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, x / 3.0).unwrap();
    let sample: f64 = normal.sample(&mut rng);
    sample.clamp(-x, x)
}

fn random_variable(fine_tune: bool) -> Expr {
    let mut rng = rand::rng();

    let fine_tune_mult = if fine_tune { 0.3 } else { 1.0 };

    let possibilities = vec![
        (Expr::Var("z".to_string()), 20.0 * fine_tune_mult),
        (Expr::Var("c".to_string()), 20.0 * fine_tune_mult),
        (
            Expr::Imag(Box::new(Expr::Var("z".to_string()))),
            1.0 * fine_tune_mult,
        ),
        (
            Expr::Real(Box::new(Expr::Var("z".to_string()))),
            1.0 * fine_tune_mult,
        ),
        (
            Expr::Imag(Box::new(Expr::Var("c".to_string()))),
            1.0 * fine_tune_mult,
        ),
        (
            Expr::Real(Box::new(Expr::Var("c".to_string()))),
            1.0 * fine_tune_mult,
        ),
        (Expr::Num(gaussian_in_range(10.0)), 20.0),
    ];

    return possibilities
        .choose_weighted(&mut rng, |item| item.1)
        .unwrap()
        .0
        .clone();
}

fn adjust_number(expr: Expr) -> Expr {
    match expr {
        Expr::Num(x) => Expr::Num(x * ((rand::rng().random_range(900..1100) as f64) / 1000.0)),
        _ => expr,
    }
}

fn is_z(expr: Expr) -> bool {
    match expr {
        Expr::Var(x) => x == "z",
        Expr::Imag(x) => is_z(*x),
        Expr::Real(x) => is_z(*x),
        _ => false,
    }
}

fn add_random_function(expr: Expr, fine_tune: bool) -> Expr {
    let mut rng = rand::rng();

    let fine_tune_mult = if fine_tune { 0.3 } else { 1.0 };

    let trig_weight = 1.0 * fine_tune_mult;
    let hyper_weight = 1.0 * fine_tune_mult;

    let possibilities = vec![
        (
            Expr::Add(Box::new(expr.clone()), Box::new(random_variable(fine_tune))),
            40.0,
        ),
        (
            Expr::Mul(Box::new(expr.clone()), Box::new(random_variable(fine_tune))),
            40.0,
        ),
        (
            Expr::Sub(Box::new(expr.clone()), Box::new(random_variable(fine_tune))),
            20.0,
        ),
        (
            {
                // We cant div by z cuz z is 0.
                let mut rand_var = random_variable(fine_tune);

                while is_z(rand_var.clone()) {
                    rand_var = random_variable(fine_tune);
                }

                Expr::Div(Box::new(expr.clone()), Box::new(rand_var))
            },
            20.0,
        ),
        (
            Expr::Pow(Box::new(expr.clone()), Box::new(random_variable(fine_tune))),
            40.0,
        ),
        //
        (Expr::Imag(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        (Expr::Real(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        (Expr::Abs(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        //
        (Expr::Exp(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        (Expr::Log(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        (Expr::Log10(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        (Expr::Sqrt(Box::new(expr.clone())), 1.0 * fine_tune_mult),
        //
        (Expr::Cos(Box::new(expr.clone())), trig_weight),
        (Expr::Sin(Box::new(expr.clone())), trig_weight),
        (Expr::Tan(Box::new(expr.clone())), trig_weight),
        (Expr::Acos(Box::new(expr.clone())), trig_weight),
        (Expr::Asin(Box::new(expr.clone())), trig_weight),
        (Expr::Atan(Box::new(expr.clone())), trig_weight),
        //
        (Expr::Cosh(Box::new(expr.clone())), hyper_weight),
        (Expr::Sinh(Box::new(expr.clone())), hyper_weight),
        (Expr::Tanh(Box::new(expr.clone())), hyper_weight),
        (Expr::Acosh(Box::new(expr.clone())), hyper_weight),
        (Expr::Asinh(Box::new(expr.clone())), hyper_weight),
        (Expr::Atanh(Box::new(expr.clone())), hyper_weight),
    ];

    possibilities
        .choose_weighted(&mut rng, |item| item.1)
        .unwrap()
        .0
        .clone()
}

fn delete_function(expr: Expr) -> Expr {
    let mut rng = rand::rng();

    match expr.clone() {
        // Binary
        Expr::Add(a, b)
        | Expr::Mul(a, b)
        | Expr::Sub(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b) => *vec![a, b].choose(&mut rng).unwrap().clone(),

        // Unary (complex)
        Expr::Imag(x)
        | Expr::Real(x)
        | Expr::Abs(x)

        // Unary (elementary)
        | Expr::Exp(x)
        | Expr::Log(x)
        | Expr::Log10(x)
        | Expr::Sqrt(x)

        // Unary (trig)
        | Expr::Cos(x)
        | Expr::Sin(x)
        | Expr::Tan(x)
        | Expr::Acos(x)
        | Expr::Asin(x)
        | Expr::Atan(x)

        // Unary (hyperbolic)
        | Expr::Cosh(x)
        | Expr::Sinh(x)
        | Expr::Tanh(x)
        | Expr::Acosh(x)
        | Expr::Asinh(x)
        | Expr::Atanh(x) => *x,

        x => x // Leave Var and Num the same
    }
}

fn switch_params(expr: Expr) -> Expr {
    match expr.clone() {
        Expr::Add(a, b) => Expr::Add(b, a),
        Expr::Mul(a, b) => Expr::Mul(b, a),
        Expr::Sub(a, b) => Expr::Sub(b, a),
        Expr::Div(a, b) => Expr::Div(b, a),
        Expr::Pow(a, b) => Expr::Pow(b, a),
        x => x,
    }
}

fn adjust_param(expr: Expr, fine_tune: bool) -> Expr {
    let fine_tune_mult = if fine_tune { 0.3 } else { 1.0 };

    let mut rng = rand::rng();
    let possibilities = vec![
        (adjust_number(expr.clone()), {
            if discriminant(&expr) == discriminant(&Expr::Num(0.0)) {
                100.0
            } else {
                0.0
            }
        }),
        (add_random_function(expr.clone(), fine_tune), 10.0),
        (delete_function(expr.clone()), 5.0 * fine_tune_mult),
        (switch_params(expr.clone()), 1.0 * fine_tune_mult),
    ];

    possibilities
        .choose_weighted(&mut rng, |item| item.1)
        .unwrap()
        .0
        .clone()
}

pub fn adjust_random_param(expr: Expr, fine_tune: bool) -> Expr {
    let mut rng = rand::rng();

    if rng.random_bool(0.3) {
        return adjust_param(expr, fine_tune);
    }

    match expr {
        // Binary
        Expr::Add(a, b) => {
            if rng.random_bool(0.5) {
                Expr::Add(Box::new(adjust_random_param(*a, fine_tune)), b)
            } else {
                Expr::Add(a, Box::new(adjust_random_param(*b, fine_tune)))
            }
        }
        Expr::Mul(a, b) => {
            if rng.random_bool(0.5) {
                Expr::Mul(Box::new(adjust_random_param(*a, fine_tune)), b)
            } else {
                Expr::Mul(a, Box::new(adjust_random_param(*b, fine_tune)))
            }
        }
        Expr::Sub(a, b) => {
            if rng.random_bool(0.5) {
                Expr::Sub(Box::new(adjust_random_param(*a, fine_tune)), b)
            } else {
                Expr::Sub(a, Box::new(adjust_random_param(*b, fine_tune)))
            }
        }
        Expr::Div(a, b) => {
            if rng.random_bool(0.5) {
                Expr::Div(Box::new(adjust_random_param(*a, fine_tune)), b)
            } else {
                Expr::Div(a, Box::new(adjust_random_param(*b, fine_tune)))
            }
        }
        Expr::Pow(a, b) => {
            if rng.random_bool(0.5) {
                Expr::Pow(Box::new(adjust_random_param(*a, fine_tune)), b)
            } else {
                Expr::Pow(a, Box::new(adjust_random_param(*b, fine_tune)))
            }
        }

        // Unary (all)
        Expr::Imag(x) => Expr::Imag(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Real(x) => Expr::Real(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Abs(x) => Expr::Abs(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Exp(x) => Expr::Exp(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Log(x) => Expr::Log(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Log10(x) => Expr::Log10(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Sqrt(x) => Expr::Sqrt(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Cos(x) => Expr::Cos(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Sin(x) => Expr::Sin(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Tan(x) => Expr::Tan(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Acos(x) => Expr::Acos(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Asin(x) => Expr::Asin(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Atan(x) => Expr::Atan(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Cosh(x) => Expr::Cosh(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Sinh(x) => Expr::Sinh(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Tanh(x) => Expr::Tanh(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Acosh(x) => Expr::Acosh(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Asinh(x) => Expr::Asinh(Box::new(adjust_random_param(*x, fine_tune))),
        Expr::Atanh(x) => Expr::Atanh(Box::new(adjust_random_param(*x, fine_tune))),

        other => adjust_param(other, fine_tune),
    }
}

pub fn get_random_formula() -> String {
    let mut x = Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Var("z".to_string())),
            Box::new(Expr::Var("z".to_string())),
        )),
        Box::new(Expr::Var("c".to_string())),
    );

    for _ in 0..100 {
        x = adjust_random_param(x, false);
    }

    let mut x_state = has_z_and_c(x.clone(), false, false);

    while !x_state.0 || !x_state.1 || check_is_fractal_monotone(from_func_notation(x.clone())) {
        x = adjust_random_param(x, false);
        x_state = has_z_and_c(x.clone(), false, false);
    }

    return from_func_notation(x);
}

fn has_z_and_c(expr: Expr, has_z: bool, has_c: bool) -> (bool, bool) {
    if has_z && has_c {
        return (true, true);
    }

    match expr {
        // Binary
        Expr::Add(a, b) => {
            let (has_z_1, has_c_1) = has_z_and_c(*a, has_z, has_c);
            let (has_z_2, has_c_2) = has_z_and_c(*b, has_z, has_c);

            (has_z_1 || has_z_2, has_c_1 || has_c_2)
        }
        Expr::Mul(a, b) => {
            let (has_z_1, has_c_1) = has_z_and_c(*a, has_z, has_c);
            let (has_z_2, has_c_2) = has_z_and_c(*b, has_z, has_c);

            (has_z_1 || has_z_2, has_c_1 || has_c_2)
        }
        Expr::Sub(a, b) => {
            let (has_z_1, has_c_1) = has_z_and_c(*a, has_z, has_c);
            let (has_z_2, has_c_2) = has_z_and_c(*b, has_z, has_c);

            (has_z_1 || has_z_2, has_c_1 || has_c_2)
        }
        Expr::Div(a, b) => {
            let (has_z_1, has_c_1) = has_z_and_c(*a, has_z, has_c);
            let (has_z_2, has_c_2) = has_z_and_c(*b, has_z, has_c);

            (has_z_1 || has_z_2, has_c_1 || has_c_2)
        }
        Expr::Pow(a, b) => {
            let (has_z_1, has_c_1) = has_z_and_c(*a, has_z, has_c);
            let (has_z_2, has_c_2) = has_z_and_c(*b, has_z, has_c);

            (has_z_1 || has_z_2, has_c_1 || has_c_2)
        }

        // Unary (all)
        Expr::Imag(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Real(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Abs(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Exp(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Log(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Log10(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Sqrt(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Cos(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Sin(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Tan(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Acos(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Asin(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Atan(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Cosh(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Sinh(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Tanh(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Acosh(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Asinh(x) => has_z_and_c(*x, has_z, has_c),
        Expr::Atanh(x) => has_z_and_c(*x, has_z, has_c),

        Expr::Num(_x) => (has_z, has_c),
        Expr::Var(x) => (has_z || x == "z", has_c || x == "c"),
    }
}

fn check_is_fractal_monotone(formula: String) -> bool {
    if create_formula_project(&formula).expect("Failed to generate Rust code") {
        compile_formula_project().expect("Failed to compile Rust code");
    }
    load_library();
    let mut fractal = Fractal::new(
        50,
        50,
        0.5,
        Complex::new(0.0, 0.0),
        100,
        32,
        None,
        crate::colors::PaletteMode::BlackAndWhite,
    );
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || make_fractal(&mut fractal, crate::fractals::FractalType::Mandelbrot))
        .unwrap();

    let bitmap = child.join().unwrap();

    for x in 0..bitmap.len() {
        for y in 0..bitmap[0].len() {
            if bitmap[x][y] != (0, 0, 0) {
                return false;
            }
        }
    }

    return true;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(e: Expr) -> String {
        from_func_notation(e)
    }

    #[test]
    fn test() {
        panic!("{}", get_random_formula());
    }

    #[test]
    fn num_and_var() {
        assert_eq!(s(Expr::Num(5.0)), "5");
        assert_eq!(s(Expr::Num(3.14)), "3.14");
        assert_eq!(s(Expr::Var("x".into())), "x");
    }

    #[test]
    fn add_simple() {
        assert_eq!(
            s(Expr::Add(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(2.0))
            )),
            "1 + 2"
        );
    }

    #[test]
    fn add_nested() {
        let e = Expr::Add(
            Box::new(Expr::Num(1.0)),
            Box::new(Expr::Add(
                Box::new(Expr::Num(2.0)),
                Box::new(Expr::Num(3.0)),
            )),
        );
        assert_eq!(s(e), "1 + 2 + 3");
    }

    #[test]
    fn ml_simple() {
        assert_eq!(
            s(Expr::Mul(
                Box::new(Expr::Num(2.0)),
                Box::new(Expr::Num(3.0))
            )),
            "2 * 3"
        );
    }

    #[test]
    fn mul_with_add_needs_parens() {
        let e = Expr::Mul(
            Box::new(Expr::Num(2.0)),
            Box::new(Expr::Add(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(3.0)),
            )),
        );
        assert_eq!(s(e), "2 * (1 + 3)");
    }

    #[test]
    fn mul_with_add_needs_parens_div() {
        let e = Expr::Mul(
            Box::new(Expr::Num(2.0)),
            Box::new(Expr::Div(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(3.0)),
            )),
        );
        assert_eq!(s(e), "2 * (1 / 3)");
    }

    #[test]
    fn sub_simple() {
        let e = Expr::Sub(Box::new(Expr::Num(5.0)), Box::new(Expr::Num(3.0)));
        assert_eq!(s(e), "5 - 3");
    }

    #[test]
    fn sub_with_add_rhs() {
        let e = Expr::Sub(
            Box::new(Expr::Num(5.0)),
            Box::new(Expr::Add(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(2.0)),
            )),
        );
        assert_eq!(s(e), "5 - (1 + 2)");
    }

    #[test]
    fn div_simple() {
        let e = Expr::Div(Box::new(Expr::Num(6.0)), Box::new(Expr::Num(2.0)));
        assert_eq!(s(e), "6 / 2");
    }

    #[test]
    fn div_with_add_rhs() {
        let e = Expr::Div(
            Box::new(Expr::Num(6.0)),
            Box::new(Expr::Add(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(2.0)),
            )),
        );
        assert_eq!(s(e), "6 / (1 + 2)");
    }

    #[test]
    fn div_with_add_lhs() {
        let e = Expr::Div(
            Box::new(Expr::Add(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(2.0)),
            )),
            Box::new(Expr::Num(3.0)),
        );
        assert_eq!(s(e), "(1 + 2) / 3");
    }

    #[test]
    fn pow_simple() {
        let e = Expr::Pow(Box::new(Expr::Num(2.0)), Box::new(Expr::Num(3.0)));
        assert_eq!(s(e), "pow(2, 3)");
    }

    #[test]
    fn pow_nested() {
        let e = Expr::Pow(
            Box::new(Expr::Add(
                Box::new(Expr::Num(1.0)),
                Box::new(Expr::Num(2.0)),
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Num(3.0)),
                Box::new(Expr::Num(4.0)),
            )),
        );
        assert_eq!(s(e), "pow(1 + 2, 3 * 4)");
    }

    #[test]
    fn complex_functions() {
        let e = Expr::Abs(Box::new(Expr::Var("z".into())));
        assert_eq!(s(e), "abs(z)");

        let e = Expr::Imag(Box::new(Expr::Var("z".into())));
        assert_eq!(s(e), "imag(z)");

        let e = Expr::Real(Box::new(Expr::Var("z".into())));
        assert_eq!(s(e), "real(z)");
    }

    #[test]
    fn elementary_functions() {
        assert_eq!(s(Expr::Exp(Box::new(Expr::Var("x".into())))), "exp(x)");
        assert_eq!(s(Expr::Log(Box::new(Expr::Var("x".into())))), "log(x)");
        assert_eq!(s(Expr::Log10(Box::new(Expr::Var("x".into())))), "log10(x)");
        assert_eq!(s(Expr::Sqrt(Box::new(Expr::Var("x".into())))), "sqrt(x)");
    }

    #[test]
    fn trig_functions() {
        assert_eq!(s(Expr::Cos(Box::new(Expr::Var("x".into())))), "cos(x)");
        assert_eq!(s(Expr::Sin(Box::new(Expr::Var("x".into())))), "sin(x)");
        assert_eq!(s(Expr::Tan(Box::new(Expr::Var("x".into())))), "tan(x)");
        assert_eq!(s(Expr::Acos(Box::new(Expr::Var("x".into())))), "acos(x)");
        assert_eq!(s(Expr::Asin(Box::new(Expr::Var("x".into())))), "asin(x)");
        assert_eq!(s(Expr::Atan(Box::new(Expr::Var("x".into())))), "atan(x)");
    }

    #[test]
    fn hyperbolic_functions() {
        assert_eq!(s(Expr::Cosh(Box::new(Expr::Var("x".into())))), "cosh(x)");
        assert_eq!(s(Expr::Sinh(Box::new(Expr::Var("x".into())))), "sinh(x)");
        assert_eq!(s(Expr::Tanh(Box::new(Expr::Var("x".into())))), "tanh(x)");
        assert_eq!(s(Expr::Acosh(Box::new(Expr::Var("x".into())))), "acosh(x)");
        assert_eq!(s(Expr::Asinh(Box::new(Expr::Var("x".into())))), "asinh(x)");
        assert_eq!(s(Expr::Atanh(Box::new(Expr::Var("x".into())))), "atanh(x)");
    }

    #[test]
    fn nested_expression_full() {
        // (2 + x) * (3 - y) / sqrt(z)
        let e = Expr::Div(
            Box::new(Expr::Mul(
                Box::new(Expr::Add(
                    Box::new(Expr::Num(2.0)),
                    Box::new(Expr::Var("x".into())),
                )),
                Box::new(Expr::Sub(
                    Box::new(Expr::Num(3.0)),
                    Box::new(Expr::Var("y".into())),
                )),
            )),
            Box::new(Expr::Sqrt(Box::new(Expr::Var("z".into())))),
        );

        assert_eq!(s(e), "(2 + x) * (3 - y) / sqrt(z)");
    }
}
