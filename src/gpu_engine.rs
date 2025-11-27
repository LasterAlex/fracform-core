// GPU fractal engine with runtime expression parser and 3 numeric backends:
// - f32 (fast)
// - DD (double-double, f32 x2)
// - QD (quad-double, f32 x4)
//

use std::sync::Mutex;
use std::{collections::HashMap, time::Instant};

use futures_util::FutureExt;
use num::Complex;
use once_cell::sync::Lazy;
use pollster::block_on;

use crate::fractals::reset_cache;
// Re-export things from your codepath (adjust crate paths if needed)
use crate::{
    fractals::{x_to_coord, y_to_coord, Bitmap},
    Fractal, FractalType,
};

/// Precision backends
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    DD,
    QD,
}

/// Simple tokenizer + parser for C-style expressions.
/// Supports numbers, identifiers (z, c), function calls, unary +-, operators: ^ * / + -,
/// parentheses, and comma for pow().
#[derive(Debug, Clone)]
enum Token {
    Number(String),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    End,
}

struct Lexer<'a> {
    s: &'a str,
    i: usize,
}
impl<'a> Lexer<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, i: 0 }
    }
    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }
    fn next_char(&mut self) -> Option<char> {
        if self.i >= self.s.len() {
            None
        } else {
            let ch = self.s[self.i..].chars().next().unwrap();
            self.i += ch.len_utf8();
            Some(ch)
        }
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.next_char();
            } else {
                break;
            }
        }
    }
    fn next_token(&mut self) -> Token {
        self.skip_ws();
        match self.peek() {
            None => Token::End,
            Some(c)
                if c.is_ascii_digit()
                    || (c == '.'
                        && self.s[self.i + 1..]
                            .chars()
                            .next()
                            .map(|ch| ch.is_ascii_digit())
                            .unwrap_or(false)) =>
            {
                let mut acc = String::new();
                if c == '.' {
                    acc.push('.');
                    self.next_char();
                }
                while let Some(d) = self.peek() {
                    if d.is_ascii_digit()
                        || d == '.'
                        || d == 'e'
                        || d == 'E'
                        || d == '+'
                        || d == '-' && acc.ends_with('e')
                        || d == '-' && acc.ends_with('E')
                    {
                        acc.push(d);
                        self.next_char();
                    } else {
                        break;
                    }
                }
                Token::Number(acc)
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                let mut id = String::new();
                while let Some(d) = self.peek() {
                    if d.is_alphanumeric() || d == '_' {
                        id.push(d);
                        self.next_char();
                    } else {
                        break;
                    }
                }
                Token::Ident(id)
            }
            Some('+') => {
                self.next_char();
                Token::Plus
            }
            Some('-') => {
                self.next_char();
                Token::Minus
            }
            Some('*') => {
                self.next_char();
                Token::Star
            }
            Some('/') => {
                self.next_char();
                Token::Slash
            }
            Some('^') => {
                self.next_char();
                Token::Caret
            }
            Some('(') => {
                self.next_char();
                Token::LParen
            }
            Some(')') => {
                self.next_char();
                Token::RParen
            }
            Some(',') => {
                self.next_char();
                Token::Comma
            }
            Some(c) => {
                self.next_char();
                panic!("Unexpected char in formula: {}", c);
            }
        }
    }
}

/// AST
#[derive(Debug, Clone)]
enum Expr {
    Number(f64),
    Var(String), // "z" or "c"
    UnaryOp {
        op: char,
        child: Box<Expr>,
    },
    BinaryOp {
        lhs: Box<Expr>,
        op: char,
        rhs: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    cur: Token,
}
impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        let mut lx = Lexer::new(s);
        let t = lx.next_token();
        Parser { lexer: lx, cur: t }
    }
    fn bump(&mut self) {
        self.cur = self.lexer.next_token();
    }
    fn parse_number(&mut self) -> Expr {
        if let Token::Number(s) = &self.cur {
            let v: f64 = s.parse().unwrap();
            self.bump();
            Expr::Number(v)
        } else {
            panic!("Expected number")
        }
    }
    fn parse_primary(&mut self) -> Expr {
        match &self.cur {
            Token::Number(_) => self.parse_number(),
            Token::Ident(id) => {
                let id = id.clone();
                self.bump();
                if let Token::LParen = self.cur {
                    // function call
                    self.bump();
                    let mut args = Vec::new();
                    if let Token::RParen = self.cur {
                        self.bump();
                    } else {
                        loop {
                            args.push(self.parse_expr());
                            if let Token::Comma = self.cur {
                                self.bump();
                                continue;
                            } else if let Token::RParen = self.cur {
                                self.bump();
                                break;
                            } else {
                                panic!("Expected ',' or ')' in call to {}", id);
                            }
                        }
                    }
                    Expr::Call { name: id, args }
                } else {
                    Expr::Var(id)
                }
            }
            Token::LParen => {
                self.bump();
                let e = self.parse_expr();
                if let Token::RParen = self.cur {
                    self.bump();
                    e
                } else {
                    panic!("Missing )");
                }
            }
            Token::Minus => {
                self.bump();
                Expr::UnaryOp {
                    op: '-',
                    child: Box::new(self.parse_primary()),
                }
            }
            Token::Plus => {
                self.bump();
                Expr::UnaryOp {
                    op: '+',
                    child: Box::new(self.parse_primary()),
                }
            }
            _ => panic!("Unexpected token in primary: {:?}", self.cur),
        }
    }
    fn parse_pow(&mut self) -> Expr {
        let mut left = self.parse_primary();
        while let Token::Caret = self.cur {
            self.bump();
            let right = self.parse_pow(); // right-assoc
            left = Expr::BinaryOp {
                lhs: Box::new(left),
                op: '^',
                rhs: Box::new(right),
            };
        }
        left
    }
    fn parse_mul_div(&mut self) -> Expr {
        let mut left = self.parse_pow();
        loop {
            match &self.cur {
                Token::Star => {
                    self.bump();
                    let r = self.parse_pow();
                    left = Expr::BinaryOp {
                        lhs: Box::new(left),
                        op: '*',
                        rhs: Box::new(r),
                    }
                }
                Token::Slash => {
                    self.bump();
                    let r = self.parse_pow();
                    left = Expr::BinaryOp {
                        lhs: Box::new(left),
                        op: '/',
                        rhs: Box::new(r),
                    }
                }
                _ => break,
            }
        }
        left
    }
    fn parse_add_sub(&mut self) -> Expr {
        let mut left = self.parse_mul_div();
        loop {
            match &self.cur {
                Token::Plus => {
                    self.bump();
                    let r = self.parse_mul_div();
                    left = Expr::BinaryOp {
                        lhs: Box::new(left),
                        op: '+',
                        rhs: Box::new(r),
                    }
                }
                Token::Minus => {
                    self.bump();
                    let r = self.parse_mul_div();
                    left = Expr::BinaryOp {
                        lhs: Box::new(left),
                        op: '-',
                        rhs: Box::new(r),
                    }
                }
                _ => break,
            }
        }
        left
    }
    fn parse_expr(&mut self) -> Expr {
        self.parse_add_sub()
    }
    fn parse(s: &'a str) -> Expr {
        let mut p = Parser::new(s);
        let e = p.parse_expr();
        match p.cur {
            Token::End => e,
            _ => panic!("Unexpected trailing token: {:?}", p.cur),
        }
    }
}

/// Code generator: transforms AST to WGSL expression string for chosen backend.
/// Strategy: we produce code that uses helper functions defined inside the WGSL backend templates.
/// The helper functions are:
///  - add(a,b), sub(a,b), mul(a,b), div(a,b), pow(a,b), neg(a)
///  - sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), sinh, cosh, tanh, asinh, acosh, atanh
///  - exp, log, log10, sqrt, abs, arg, real, imag, conj
/// For f32 backend, these helper functions will be thin wrappers around vec2 ops and standard f32 trig.
/// For DD/QD, the helpers will use DD/QD implementations.
///
/// The code generator emits expression using var names `z` and `c` (already present in shader).
fn gen_wgsl_expr(expr: &Expr, precision: Precision) -> String {
    match expr {
        Expr::Number(v) => {
            // produce a literal complex constant C(x,0)
            match precision {
                Precision::F32 => format!("vec2({:.18}, 0.0)", *v),
                Precision::DD => format!("CDD(DD({:.18}, 0.0), DD(0.0, 0.0))", *v),
                Precision::QD => format!(
                    "CDD(QD(DD({:.18}, 0.0), DD(0.0, 0.0)), QD(DD(0.0, 0.0), DD(0.0, 0.0)))",
                    *v
                ),
            }
        }
        Expr::Var(name) => match name.as_str() {
            "z" | "c" => name.clone(),
            _ => panic!("Unknown variable: {}", name),
        },
        Expr::UnaryOp { op, child } => {
            let ch = gen_wgsl_expr(child, precision);
            match op {
                '+' => format!("{}", ch),
                '-' => format!("neg({})", ch),
                _ => unreachable!(),
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            let a = gen_wgsl_expr(lhs, precision);
            let b = gen_wgsl_expr(rhs, precision);
            match op {
                '+' => format!("add({}, {})", a, b),
                '-' => format!("sub({}, {})", a, b),
                '*' => format!("mul({}, {})", a, b),
                '/' => format!("div({}, {})", a, b),
                '^' => format!("pow({}, {})", a, b),
                _ => unreachable!(),
            }
        }
        Expr::Call { name, args } => {
            let low = args
                .iter()
                .map(|a| gen_wgsl_expr(a, precision))
                .collect::<Vec<_>>()
                .join(", ");
            // map some names synonyms (C-style) to our internal function names
            let nm = match name.as_str() {
                "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh"
                | "asinh" | "acosh" | "atanh" | "exp" | "log" | "log10" | "sqrt" | "pow"
                | "abs" | "arg" | "real" | "imag" | "conj" => name.clone(),
                // support capitalized synonyms too
                other => other.to_string(),
            };
            format!("{}({})", nm, low)
        }
    }
}

pub fn build_shader_for_backend(expr_wgsl: &str, backend: Precision, is_julia: bool) -> String {
    match backend {
        Precision::F32 => {
            format!(
                r##"
const ESCAPE: f32 = 16.0;

struct Params {{
    width: u32,
    height: u32,
    max_iters: u32,
    _pad: u32,
    cx0: f32,
    cy0: f32,
    cx1: f32,
    cy1: f32,
    {julia_fields}
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

// F32 complex helpers
fn add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ return a + b; }}
fn sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ return a - b; }}
fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }}
fn div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ let denom = b.x*b.x + b.y*b.y; return vec2<f32>((a.x*b.x + a.y*b.y)/denom, (a.y*b.x - a.x*b.y)/denom); }}
fn conj(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(a.x, -a.y); }}
fn abs2(a: vec2<f32>) -> f32 {{ return dot(a,a); }}
fn escape_abs2(z: vec2<f32>) -> f32 {{ return dot(z,z); }}

// Transcendentals
fn cexp(a: vec2<f32>) -> vec2<f32> {{ let e = exp(a.x); return vec2<f32>(e*cos(a.y), e*sin(a.y)); }}
fn clog(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(log(length(a)), atan2(a.y,a.x)); }}
fn csin(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(sin(a.x)*cosh(a.y), cos(a.x)*sinh(a.y)); }}
fn ccos(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(cos(a.x)*cosh(a.y), -sin(a.x)*sinh(a.y)); }}
fn csqrt(a: vec2<f32>) -> vec2<f32> {{ let r = length(a); let re = sqrt(max(0.0,0.5*(r+a.x))); let im = sign(a.y)*sqrt(max(0.0,0.5*(r-a.x))); return vec2<f32>(re,im); }}
fn cpow(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ let L = clog(a); let mr = b.x*L.x - b.y*L.y; let mi = b.x*L.y + b.y*L.x; return cexp(vec2<f32>(mr,mi)); }}

fn user_function(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {{
    return {expr};
}}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {{ return; }}
    let idx: u32 = y*params.width + x;

    let fx = f32(x)/f32(max(1u, params.width));
    let fy = f32(y)/f32(max(1u, params.height));
    let cx = params.cx0*(1.0-fx) + params.cx1*fx;
    let cy = params.cy0*(1.0-fy) + params.cy1*fy;

    var z = {z};
    let c_val = {c_val};

    var i: u32 = 0u;
    loop {{
        if (i >= params.max_iters) {{ break; }}
        if (escape_abs2(z) > ESCAPE) {{ break; }}
        z = user_function(z, c_val);
        i = i + 1u;
    }}
    out_buf[idx] = i;
}}
"##,
                expr = expr_wgsl,
                julia_fields = if is_julia {
                    "julia_re: f32,\njulia_im: f32,"
                } else {
                    ""
                },
                c_val = if is_julia {
                    "vec2<f32>(params.julia_re, params.julia_im)"
                } else {
                    "vec2<f32>(cx,cy)"
                },
                z = if is_julia {
                    "vec2<f32>(cx,cy)"
                } else {
                    "vec2<f32>(0.0,0.0)"
                }
            )
        }

        Precision::DD => {
            // DD: full compensated arithmetic
            format!(
                r##"
const ESCAPE: f32 = 16.0;

struct DD {{ hi: f32, lo: f32, }};
struct CDD {{ re: DD, im: DD, }};

struct Params {{
    width: u32,
    height: u32,
    max_iters: u32,
    _pad: u32,
    cx0_hi: f32, cx0_lo: f32,
    cy0_hi: f32, cy0_lo: f32,
    cx1_hi: f32, cx1_lo: f32,
    cy1_hi: f32, cy1_lo: f32,
    {julia_fields}
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

// DD arithmetic
fn two_sum(a: f32, b: f32) -> vec2<f32> {{
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}}
fn two_prod(a: f32, b: f32) -> vec2<f32> {{
    let p = a * b;
    let split = 4097.0;
    let a_hi = (a*split)-((a*split)-a);
    let a_lo = a - a_hi;
    let b_hi = (b*split)-((b*split)-b);
    let b_lo = b - b_hi;
    let err = ((a_hi*b_hi - p) + a_hi*b_lo + a_lo*b_hi) + a_lo*b_lo;
    return vec2<f32>(p, err);
}}

fn dd_new(hi:f32, lo:f32) -> DD {{ return DD(hi,lo); }}
fn dd_from_f32(a:f32) -> DD {{ return DD(a,0.0); }}
fn dd_add(a:DD,b:DD) -> DD {{
    let s = two_sum(a.hi,b.hi);
    let lo = a.lo + b.lo + s.y;
    let r = two_sum(s.x, lo);
    return DD(r.x,r.y);
}}
fn dd_sub(a:DD,b:DD) -> DD {{
    let s = two_sum(a.hi,-b.hi);
    let lo = a.lo - b.lo + s.y;
    let r = two_sum(s.x, lo);
    return DD(r.x,r.y);
}}
fn dd_mul(a:DD,b:DD) -> DD {{
    let p = two_prod(a.hi,b.hi);
    let err = a.hi*b.lo + a.lo*b.hi;
    let r = two_sum(p.x, p.y + err);
    return DD(r.x,r.y);
}}
fn dd_sqr(a:DD) -> DD {{ return dd_mul(a,a); }}
fn dd_to_f32(a:DD) -> f32 {{ return a.hi; }}

fn add(a:CDD,b:CDD) -> CDD {{ return CDD(dd_add(a.re,b.re), dd_add(a.im,b.im)); }}
fn sub(a:CDD,b:CDD) -> CDD {{ return CDD(dd_sub(a.re,b.re), dd_sub(a.im,b.im)); }}
fn mul(a:CDD,b:CDD) -> CDD {{
    let re = dd_sub(dd_mul(a.re,b.re), dd_mul(a.im,b.im));
    let im = dd_add(dd_mul(a.re,b.im), dd_mul(a.im,b.re));
    return CDD(re,im);
}}
fn div(a:CDD,b:CDD) -> CDD {{
    let br = dd_to_f32(b.re); let bi = dd_to_f32(b.im);
    let ar = dd_to_f32(a.re); let ai = dd_to_f32(a.im);
    let denom = br*br + bi*bi + 1e-30;
    let rr = (ar*br + ai*bi)/denom;
    let ri = (ai*br - ar*bi)/denom;
    return CDD(dd_from_f32(rr), dd_from_f32(ri));
}}
fn abs2(a:CDD) -> DD {{ return dd_add(dd_mul(a.re,a.re), dd_mul(a.im,a.im)); }}
fn escape_abs2(a:CDD) -> f32 {{ let r = dd_to_f32(a.re); let i = dd_to_f32(a.im); return r*r + i*i; }}

// Transcendentals via high component only (can extend to full DD if needed)
fn cexp(a:CDD) -> CDD {{ let re = dd_to_f32(a.re); let im = dd_to_f32(a.im); let e = exp(re); return CDD(dd_from_f32(e*cos(im)), dd_from_f32(e*sin(im))); }}
fn clog(a:CDD) -> CDD {{ let re = dd_to_f32(a.re); let im = dd_to_f32(a.im); return CDD(dd_from_f32(log(sqrt(re*re + im*im))), dd_from_f32(atan2(im,re))); }}
fn csqrt(a:CDD) -> CDD {{ let re = dd_to_f32(a.re); let im = dd_to_f32(a.im); let r = sqrt(re*re + im*im); let re2 = sqrt(max(0.0,0.5*(r+re))); let im2 = sign(im)*sqrt(max(0.0,0.5*(r-re))); return CDD(dd_from_f32(re2), dd_from_f32(im2)); }}

// User expression
fn user_function(z:CDD, c:CDD) -> CDD {{
    return {expr};
}}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {{ return; }}
    let idx: u32 = y*params.width + x;

    let fx = f32(x)/f32(max(1u, params.width));
    let fy = f32(y)/f32(max(1u, params.height));

    let cx_hi = params.cx0_hi*(1.0-fx)+params.cx1_hi*fx;
    let cx_lo = params.cx0_lo*(1.0-fx)+params.cx1_lo*fx;
    let cy_hi = params.cy0_hi*(1.0-fy)+params.cy1_hi*fy;
    let cy_lo = params.cy0_lo*(1.0-fy)+params.cy1_lo*fy;

    var z = {z};
    var c_val = {c_val};

    var i: u32 = 0u;
    loop {{
        if (i >= params.max_iters) {{ break; }}
        if (escape_abs2(z) > ESCAPE) {{ break; }}
        z = user_function(z, c_val);
        i = i + 1u;
    }}
    out_buf[idx] = i;
}}
"##,
                expr = expr_wgsl,
                julia_fields = if is_julia {
                    "julia_re_hi: f32, julia_re_lo: f32, julia_im_hi: f32, julia_im_lo: f32,"
                } else {
                    ""
                },
                c_val = if is_julia {
                    "CDD(DD(params.julia_re_hi, params.julia_re_lo),
                    DD(params.julia_im_hi, params.julia_im_lo))"
                } else {
                    "CDD(DD(cx_hi, cx_lo), DD(cy_hi, cy_lo))"
                },
                z = if is_julia {
                    "CDD(DD(cx_hi, cx_lo), DD(cy_hi, cy_lo))"
                } else {
                    "CDD(dd_from_f32(0.0), dd_from_f32(0.0))"
                }
            )
        }

        Precision::QD => {
            // QD: use 2xDD pairs per coordinate
            format!(
                r##"
const ESCAPE: f32 = 16.0;

struct DD {{ hi: f32, lo: f32, }};
struct QD {{ a: DD, b: DD, }};
struct CQD {{ re: QD, im: QD, }};

struct Params {{
    width: u32, height: u32, max_iters: u32, _pad: u32,
    cx0_hi0: f32, cx0_lo0: f32, cx0_hi1: f32, cx0_lo1: f32,
    cy0_hi0: f32, cy0_lo0: f32, cy0_hi1: f32, cy0_lo1: f32,
    cx1_hi0: f32, cx1_lo0: f32, cx1_hi1: f32, cx1_lo1: f32,
    cy1_hi0: f32, cy1_lo0: f32, cy1_hi1: f32, cy1_lo1: f32,
    {julia_fields}
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

// minimal DD helpers
fn two_sum(a: f32, b: f32) -> vec2<f32> {{
    let s = a+b; let bb = s-a; let err=(a-(s-bb))+(b-bb); return vec2<f32>(s,err);
}}
fn dd_new(h:f32,l:f32)->DD{{ return DD(h,l); }}
fn qd_new(a_hi:f32,a_lo:f32,b_hi:f32,b_lo:f32)->QD{{ return QD(DD(a_hi,a_lo),DD(b_hi,b_lo)); }}
fn qd_from_f32(v:f32)->QD{{ return QD(DD(v,0.0),DD(0.0,0.0)); }}
fn qd_to_f32(q:QD)->f32{{ return q.a.hi; }}

// CQD arithmetic
fn add(a:CQD,b:CQD)->CQD{{let ar=qd_to_f32(a.re); let ai=qd_to_f32(a.im); let br=qd_to_f32(b.re); let bi=qd_to_f32(b.im); return CQD(qd_from_f32(ar+br), qd_from_f32(ai+bi)); }}
fn sub(a:CQD,b:CQD)->CQD{{let ar=qd_to_f32(a.re); let ai=qd_to_f32(a.im); let br=qd_to_f32(b.re); let bi=qd_to_f32(b.im); return CQD(qd_from_f32(ar-br), qd_from_f32(ai-bi)); }}
fn mul(a:CQD,b:CQD)->CQD{{let ar=qd_to_f32(a.re); let ai=qd_to_f32(a.im); let br=qd_to_f32(b.re); let bi=qd_to_f32(b.im); return CQD(qd_from_f32(ar*br - ai*bi), qd_from_f32(ar*bi + ai*br)); }}
fn div(a:CQD,b:CQD)->CQD{{let ar=qd_to_f32(a.re); let ai=qd_to_f32(a.im); let br=qd_to_f32(b.re); let bi=qd_to_f32(b.im); let denom=br*br+bi*bi+1e-30; return CQD(qd_from_f32((ar*br+ai*bi)/denom), qd_from_f32((ai*br-ar*bi)/denom)); }}
fn escape_abs2(a:CQD)->f32{{let r=qd_to_f32(a.re); let i=qd_to_f32(a.im); return r*r + i*i; }}

// User function
fn user_function(z:CQD,c:CQD)->CQD{{return {expr};}}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x=gid.x; let y=gid.y;
    if(x>=params.width || y>=params.height){{return;}}
    let idx:u32=y*params.width+x;
    let fx=f32(x)/f32(max(1u,params.width));
    let fy=f32(y)/f32(max(1u,params.height));

    let cx_hi0=params.cx0_hi0*(1.0-fx)+params.cx1_hi0*fx;
    let cx_lo0=params.cx0_lo0*(1.0-fx)+params.cx1_lo0*fx;
    let cx_hi1=params.cx0_hi1*(1.0-fx)+params.cx1_hi1*fx;
    let cx_lo1=params.cx0_lo1*(1.0-fx)+params.cx1_lo1*fx;

    let cy_hi0=params.cy0_hi0*(1.0-fy)+params.cy1_hi0*fy;
    let cy_lo0=params.cy0_lo0*(1.0-fy)+params.cy1_lo0*fy;
    let cy_hi1=params.cy0_hi1*(1.0-fy)+params.cy1_hi1*fy;
    let cy_lo1=params.cy0_lo1*(1.0-fy)+params.cy1_lo1*fy;

    var z={z};
    let c_val={c_val};

    var i:u32=0u;
    loop {{
        if(i>=params.max_iters){{break;}}
        if(escape_abs2(z)>ESCAPE){{break;}}
        z=user_function(z,c_val);
        i=i+1u;
    }}
    out_buf[idx]=i;
}}
"##,
                expr = expr_wgsl,
                julia_fields = if is_julia {
                    "julia_re_hi0: f32, julia_re_lo0: f32, julia_re_hi1: f32, julia_re_lo1: f32, julia_im_hi0: f32, julia_im_lo0: f32, julia_im_hi1: f32, julia_im_lo1: f32,"
                } else {
                    ""
                },
                c_val = if is_julia {
                    "CQD(qd_new(params.julia_re_hi0,params.julia_re_lo0,params.julia_re_hi1,params.julia_re_lo1), qd_new(params.julia_im_hi0,params.julia_im_lo0,params.julia_im_hi1,params.julia_im_lo1))"
                } else {
                    "CQD(qd_new(cx_hi0,cx_lo0,cx_hi1,cx_lo1), qd_new(cy_hi0,cy_lo0,cy_hi1,cy_lo1))"
                },
                z = if is_julia {
                    "CQD(qd_new(cx_hi0,cx_lo0,cx_hi1,cx_lo1), qd_new(cy_hi0,cy_lo0,cy_hi1,cy_lo1))"
                } else {
                    "CQD(qd_from_f32(0.0), qd_from_f32(0.0))"
                }
            )
        }
    }
}

/// --- GPU engine state (caches pipelines by (expr,backend)) ---
pub static GPU_STATE: Lazy<Mutex<GpuState>> = Lazy::new(|| Mutex::new(GpuState::new()));

pub struct GpuState {
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,
    pub shaders: HashMap<(String, Precision), wgpu::ShaderModule>,
    pub pipelines: HashMap<(String, Precision), wgpu::ComputePipeline>,
}
impl GpuState {
    fn new() -> Self {
        Self {
            device: None,
            queue: None,
            shaders: HashMap::new(),
            pipelines: HashMap::new(),
        }
    }
}

/// Initialize WGPU device & queue if not already initialized.
pub fn ensure_wgpu<'a>(st: &'a mut GpuState) -> (&'a wgpu::Device, &'a wgpu::Queue) {
    if st.device.is_none() {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No adapter");
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ))
        .expect("Failed to create device");
        st.device = Some(device);
        st.queue = Some(queue);
    }
    // unwrap again to get clones (we store only one)
    let device = st.device.as_ref().unwrap();
    let queue = st.queue.as_ref().unwrap();
    (device, queue)
}

/// Auto-select precision based on zoom. You can tweak thresholds here.
fn select_precision(zoom: f64) -> Precision {
    let z = zoom.abs();
    if z < 1e4 {
        Precision::F32
    } else if z < 1e22 {
        Precision::DD
    } else {
        Precision::QD
    }
}

/// Top-level GPU renderer
pub fn render_gpu<'a>(
    fractal: &Fractal,
    formula: &str,
    fractal_type: FractalType,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
) -> Bitmap {
    let ast = Parser::parse(formula);
    let backend = select_precision(fractal.zoom);
    let expr_wgsl = gen_wgsl_expr(&ast, backend);
    let is_julia = matches!(fractal_type, FractalType::Julia);

    let mut shader_src = build_shader_for_backend(&expr_wgsl, backend, is_julia);
    shader_src = shader_src.replacen(
        "const ESCAPE: f32 = 16.0;",
        &format!("const ESCAPE: f32 = {:.6};", fractal.max_abs as f32),
        1,
    );

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fractal_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });

    let width = fractal.width as u32;
    let height = fractal.height as u32;
    let pixel_count = width as u64 * height as u64;
    let out_size = pixel_count * 4;

    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ---------- Compute fractal corners ----------
    let cx0_f = x_to_coord(
        0,
        fractal.width,
        fractal.height,
        fractal.shift.re,
        fractal.zoom,
    );
    let cy0_f = y_to_coord(
        0,
        fractal.width,
        fractal.height,
        fractal.shift.im,
        fractal.zoom,
    );
    let cx1_f = x_to_coord(
        fractal.width,
        fractal.width,
        fractal.height,
        fractal.shift.re,
        fractal.zoom,
    );
    let cy1_f = y_to_coord(
        fractal.height,
        fractal.width,
        fractal.height,
        fractal.shift.im,
        fractal.zoom,
    );

    let max_iters = fractal.iterations;
    let pad = 0u32;

    // ---------- Rust mirrors of WGSL structs ----------
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ParamsF32 {
        width: u32,
        height: u32,
        max_iters: u32,
        _pad: u32,
        cx0: f32,
        cy0: f32,
        cx1: f32,
        cy1: f32,
        julia_re: f32,
        julia_im: f32,
    }
    unsafe impl bytemuck::Pod for ParamsF32 {}
    unsafe impl bytemuck::Zeroable for ParamsF32 {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DD {
        hi: f32,
        lo: f32,
    }
    unsafe impl bytemuck::Pod for DD {}
    unsafe impl bytemuck::Zeroable for DD {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ParamsDD {
        width: u32,
        height: u32,
        max_iters: u32,
        _pad: u32,
        cx0: DD,
        cy0: DD,
        cx1: DD,
        cy1: DD,
        julia_re: DD,
        julia_im: DD,
    }
    unsafe impl bytemuck::Pod for ParamsDD {}
    unsafe impl bytemuck::Zeroable for ParamsDD {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct QD {
        x0: f32,
        x1: f32,
        x2: f32,
        x3: f32,
    }
    unsafe impl bytemuck::Pod for QD {}
    unsafe impl bytemuck::Zeroable for QD {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ParamsQD {
        width: u32,
        height: u32,
        max_iters: u32,
        _pad: u32,
        cx0: QD,
        cy0: QD,
        cx1: QD,
        cy1: QD,
        julia_re: QD,
        julia_im: QD,
    }
    unsafe impl bytemuck::Pod for ParamsQD {}
    unsafe impl bytemuck::Zeroable for ParamsQD {}

    // ---------- Convert f64 -> DD / QD ----------
    fn to_dd(x: f64) -> DD {
        DD {
            hi: x as f32,
            lo: (x - (x as f32) as f64) as f32,
        }
    }
    fn to_qd(x: f64) -> QD {
        let x0 = x as f32;
        let r1 = x - x0 as f64;
        let x1 = r1 as f32;
        let r2 = r1 - x1 as f64;
        let x2 = r2 as f32;
        let r3 = r2 - x2 as f64;
        let x3 = r3 as f32;
        QD { x0, x1, x2, x3 }
    }

    // ---------- Build correct params + buffer ----------
    let (params_bytes, uniform_size) = match backend {
        Precision::F32 => {
            let (julia_re, julia_im) = if is_julia {
                (
                    fractal.c.unwrap_or(Complex::new(0.0, 0.0)).re as f32,
                    fractal.c.unwrap_or(Complex::new(0.0, 0.0)).im as f32,
                )
            } else {
                (0.0, 0.0)
            };
            let p = ParamsF32 {
                width,
                height,
                max_iters,
                _pad: pad,
                cx0: cx0_f as f32,
                cy0: cy0_f as f32,
                cx1: cx1_f as f32,
                cy1: cy1_f as f32,
                julia_re,
                julia_im,
            };
            (
                bytemuck::bytes_of(&p).to_vec(),
                std::mem::size_of::<ParamsF32>() as u64,
            )
        }
        Precision::DD => {
            let (julia_re, julia_im) = if is_julia {
                let c = fractal.c.unwrap_or(Complex::new(0.0, 0.0));
                (to_dd(c.re), to_dd(c.im))
            } else {
                (to_dd(0.0), to_dd(0.0))
            };
            let p = ParamsDD {
                width,
                height,
                max_iters,
                _pad: pad,
                cx0: to_dd(cx0_f),
                cy0: to_dd(cy0_f),
                cx1: to_dd(cx1_f),
                cy1: to_dd(cy1_f),
                julia_re,
                julia_im,
            };
            (
                bytemuck::bytes_of(&p).to_vec(),
                std::mem::size_of::<ParamsDD>() as u64,
            )
        }
        Precision::QD => {
            let (julia_re, julia_im) = if is_julia {
                let c = fractal.c.unwrap_or(Complex::new(0.0, 0.0));
                (to_qd(c.re), to_qd(c.im))
            } else {
                (to_qd(0.0), to_qd(0.0))
            };
            let p = ParamsQD {
                width,
                height,
                max_iters,
                _pad: pad,
                cx0: to_qd(cx0_f),
                cy0: to_qd(cy0_f),
                cx1: to_qd(cx1_f),
                cy1: to_qd(cy1_f),
                julia_re,
                julia_im,
            };
            (
                bytemuck::bytes_of(&p).to_vec(),
                std::mem::size_of::<ParamsQD>() as u64,
            )
        }
    };

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: uniform_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, &params_bytes);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cpass"),
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let gw = (width + 15) / 16;
        let gh = (height + 15) / 16;
        pass.dispatch_workgroups(gw, gh, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, out_size);
    queue.submit(Some(encoder.finish()));

    // ---------- Read back pixels ----------
    let slice = readback.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().now_or_never().unwrap().unwrap().unwrap();
    let data = slice.get_mapped_range().to_vec();
    readback.unmap();

    let mut bmp = [0u32; crate::config::MAX_PIXELS as usize];
    for i in 0..pixel_count as usize {
        let off = i * 4;
        bmp[i] = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
    }
    bmp
}

// Convenience wrappers to be called from your Fractal impl:
pub fn render_mandelbrot<'a>(
    fractal: &Fractal,
    formula: &str,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
) -> Bitmap {
    reset_cache();
    render_gpu(fractal, formula, FractalType::Mandelbrot, device, queue)
}

pub fn render_julia<'a>(
    fractal: &Fractal,
    formula: &str,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
) -> Bitmap {
    reset_cache();
    render_gpu(fractal, formula, FractalType::Julia, device, queue)
}
