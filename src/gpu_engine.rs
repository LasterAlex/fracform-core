use std::sync::Mutex;
use std::{collections::HashMap};

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

fn gen_wgsl_expr(expr: &Expr) -> String {
    match expr {
        Expr::Number(v) => {
            format!("vec2({:.18}, 0.0)", *v)
        }
        Expr::Var(name) => match name.as_str() {
            "z" | "c" => name.clone(),
            _ => panic!("Unknown variable: {}", name),
        },
        Expr::UnaryOp { op, child } => {
            let ch = gen_wgsl_expr(child);
            match op {
                '+' => format!("{}", ch),
                '-' => format!("neg({})", ch),
                _ => unreachable!(),
            }
        }
        Expr::BinaryOp { lhs, op, rhs } => {
            let a = gen_wgsl_expr(lhs);
            let b = gen_wgsl_expr(rhs);
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
                .map(|a| gen_wgsl_expr(a))
                .collect::<Vec<_>>()
                .join(", ");
            // map some names synonyms (C-style) to our internal function names
            let nm = match name.as_str() {
                "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh"
                | "asinh" | "acosh" | "atanh" | "exp" | "log" | "log10" | "sqrt" | "pow"
                | "abs" | "arg" | "real" | "imag" | "conj" => format!("c_{}", name.clone()),
                // support capitalized synonyms too
                other => other.to_string(),
            };
            format!("{}({})", nm, low)
        }
    }
}

pub fn build_shader_for_backend(expr_wgsl: &str, is_julia: bool) -> String {
    build_f32_shader(expr_wgsl, is_julia)
}

fn build_f32_shader(expr_wgsl: &str, is_julia: bool) -> String {
    format!(
        r#"
const ESCAPE: f32 = 16.0;
const PI: f32 = 3.14159265359;

struct Params {{
    width: u32, height: u32, max_iters: u32, _pad: u32,
    cx0: f32, cy0: f32, cx1: f32, cy1: f32,
    {julia_fields}
}};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

fn add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ return a + b; }}
fn sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ return a - b; }}
fn neg(a: vec2<f32>) -> vec2<f32> {{ return -a; }}
fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{ 
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); 
}}
fn div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    let denom = b.x*b.x + b.y*b.y + 1e-30;
    return vec2<f32>((a.x*b.x + a.y*b.y)/denom, (a.y*b.x - a.x*b.y)/denom);
}}
fn c_conj(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(a.x, -a.y); }}
fn c_real(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(a.x, 0.0); }}
fn c_imag(a: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(a.y, 0.0); }}
fn abs2(a: vec2<f32>) -> f32 {{ return dot(a, a); }}
fn escape_abs2(a: vec2<f32>) -> f32 {{ return dot(a,a); }}

fn c_abs(z: vec2<f32>) -> vec2<f32> {{ 
    let mag_sq = z.x * z.x + z.y * z.y;
    return vec2<f32>(sqrt(mag_sq), 0.0); // built-in sqrt(f32)
}}
fn arg(z: vec2<f32>) -> f32 {{ return atan2(z.y, z.x); }}
fn c_exp(z: vec2<f32>) -> vec2<f32> {{
    let e_x = exp(z.x); // built-in exp(f32)
    return vec2<f32>(e_x * cos(z.y), e_x * sin(z.y));
}}
fn c_log(z: vec2<f32>) -> vec2<f32> {{ 
    let mag = c_abs(z).x;
    return vec2<f32>(log(mag), arg(z)); // built-in log(f32)
}}
fn c_sqrt(z: vec2<f32>) -> vec2<f32> {{
    let r = c_abs(z).x;
    let theta = arg(z);
    let sqrt_r = sqrt(r); // built-in sqrt(f32)
    let half_theta = theta / 2.0;
    return vec2<f32>(sqrt_r * cos(half_theta), sqrt_r * sin(half_theta));
}}
fn c_pow(z: vec2<f32>, p: vec2<f32>) -> vec2<f32> {{ return c_exp(mul(p, c_log(z))); }}
fn c_sin(z: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}}
fn c_cos(z: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}}
fn c_tan(z: vec2<f32>) -> vec2<f32> {{ return div(c_sin(z), c_cos(z)); }}
fn c_sinh(z: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y));
}}
fn c_cosh(z: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(cosh(z.x) * cos(z.y), sinh(z.x) * sin(z.y));
}}
fn c_tanh(z: vec2<f32>) -> vec2<f32> {{ return div(c_sinh(z), c_cosh(z)); }}

//
// INVERSE TRIGONOMETRIC COMPLEX FUNCTIONS
//

fn c_asin(z: vec2<f32>) -> vec2<f32> {{
    // asin(z) = -i * log( i*z + sqrt(1 - z*z) )
    let i_z = vec2<f32>(-z.y, z.x);        // i*z
    let one = vec2<f32>(1.0, 0.0);
    let inside_sqrt = sub(one, mul(z, z));
    let root = c_sqrt(inside_sqrt);
    let sum = add(i_z, root);
    let ln = c_log(sum);
    return vec2<f32>(ln.y, -ln.x);         // -i * ln
}}

fn c_acos(z: vec2<f32>) -> vec2<f32> {{
    // acos(z) = π/2 - asin(z)
    let a = c_asin(z);
    return vec2<f32>(PI * 0.5 - a.x, -a.y);
}}

fn c_atan(z: vec2<f32>) -> vec2<f32> {{
    let i = vec2<f32>(0.0, 1.0);
    let num = add(i, z);
    let den = sub(i, z);
    let frac = div(num, den);
    let ln = c_log(frac);
    return vec2<f32>(-0.5 * ln.y, 0.5 * ln.x);  // (i/2)*ln
}}

//
// INVERSE HYPERBOLIC COMPLEX FUNCTIONS
//

fn c_asinh(z: vec2<f32>) -> vec2<f32> {{
    // asinh(z) = log(z + sqrt(z*z + 1))
    let one = vec2<f32>(1.0, 0.0);
    let inside = add(mul(z, z), one);
    let root = c_sqrt(inside);
    return c_log(add(z, root));
}}

fn c_acosh(z: vec2<f32>) -> vec2<f32> {{
    // acosh(z) = log(z + sqrt(z+1) * sqrt(z-1))
    let one = vec2<f32>(1.0, 0.0);
    let zp = add(z, one);
    let zm = sub(z, one);
    let root = mul(c_sqrt(zp), c_sqrt(zm));
    return c_log(add(z, root));
}}

fn c_atanh(z: vec2<f32>) -> vec2<f32> {{
    // atanh(z) = 0.5 * log( (1+z) / (1-z) )
    let one = vec2<f32>(1.0, 0.0);
    let num = add(one, z);
    let den = sub(one, z);
    let frac = div(num, den);
    let ln = c_log(frac);
    return vec2<f32>(0.5 * ln.x, 0.5 * ln.y);
}}

//
// LOG BASE 10
//

fn c_log10(z: vec2<f32>) -> vec2<f32> {{
    let natural = c_log(z);
    let inv_ln10 = 1.0 / log(10.0);
    return vec2<f32>(natural.x * inv_ln10, natural.y * inv_ln10);
}}

fn user_function(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {{ return {expr}; }}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {{ return; }}
    let idx: u32 = x * params.height + y;

    let fx = f32(x) / f32(max(1u, params.width - 1u));
    let fy = f32(y) / f32(max(1u, params.height - 1u));
    let cx = params.cx0 * (1.0 - fx) + params.cx1 * fx;
    let cy = params.cy0 * (1.0 - fy) + params.cy1 * fy;

    var z: vec2<f32> = {z_init};
    let c_val: vec2<f32> = {c_init};

    var i: u32 = 0u;
    loop {{
        if (i >= params.max_iters) {{ break; }}
        if (escape_abs2(z) > ESCAPE) {{ break; }}
        z = user_function(z, c_val);
        i = i + 1u;
    }}
    out_buf[idx] = i;
}}
"#,
        expr = expr_wgsl,
        julia_fields = if is_julia {
            "julia_re: f32, julia_im: f32,"
        } else {
            ""
        },
        z_init = if is_julia {
            "vec2<f32>(cx, cy)"
        } else {
            "vec2<f32>(0.0, 0.0)"
        },
        c_init = if is_julia {
            "vec2<f32>(params.julia_re, params.julia_im)"
        } else {
            "vec2<f32>(cx, cy)"
        }
    )
}

/// --- GPU engine state (caches pipelines by (expr,backend)) ---
pub static GPU_STATE: Lazy<Mutex<GpuState>> = Lazy::new(|| Mutex::new(GpuState::new()));

pub struct GpuState {
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,
    pub shaders: HashMap<String, wgpu::ShaderModule>,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
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

/// Top-level GPU renderer
pub fn render_gpu<'a>(
    fractal: &Fractal,
    formula: &str,
    fractal_type: FractalType,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
) -> Bitmap {
    let ast = Parser::parse(formula);
    let expr_wgsl = gen_wgsl_expr(&ast);
    let is_julia = matches!(fractal_type, FractalType::Julia);

    let mut shader_src = build_shader_for_backend(&expr_wgsl, is_julia);
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

    // ---------- Build correct params + buffer ----------
    let (params_bytes, uniform_size) = {
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
