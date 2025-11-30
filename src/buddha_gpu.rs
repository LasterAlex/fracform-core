use num::Complex;
use pollster::block_on;
use std::time::Instant;
use wgpu;

use crate::{
    fractals::{reset_cache, x_to_coord, y_to_coord, Bitmap},
    Fractal, FractalType,
};

use super::gpu_engine::{ensure_wgpu, gen_wgsl_expr, Parser, GPU_STATE};

/// Build the Buddhabrot compute shader
fn build_buddhabrot_shader(
    expr_wgsl: &str,
    is_antibuddhabrot: bool,
    max_abs: u32,
    max_iters: u32,
) -> String {
    // Limit trajectory length based on iterations
    let traj_size = (max_iters.min(4096)) as usize;

    format!(
        r#"
const ESCAPE: f32 = {escape:.6};
const PI: f32 = 3.14159265359;
const MAX_TRAJECTORY: u32 = {traj_size}u;

struct Params {{
    width: u32,
    height: u32,
    max_iters: u32,
    num_points: u32,
    cx0: f32,
    cy0: f32,
    cx1: f32,
    cy1: f32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bitmap: array<atomic<u32>>;

// Complex arithmetic functions
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
    return vec2<f32>(sqrt(mag_sq), 0.0);
}}
fn arg(z: vec2<f32>) -> f32 {{ return atan2(z.y, z.x); }}
fn c_exp(z: vec2<f32>) -> vec2<f32> {{
    let e_x = exp(z.x);
    return vec2<f32>(e_x * cos(z.y), e_x * sin(z.y));
}}
fn c_log(z: vec2<f32>) -> vec2<f32> {{ 
    let mag = c_abs(z).x;
    return vec2<f32>(log(mag), arg(z));
}}
fn c_sqrt(z: vec2<f32>) -> vec2<f32> {{
    let r = c_abs(z).x;
    let theta = arg(z);
    let sqrt_r = sqrt(r);
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

fn c_asin(z: vec2<f32>) -> vec2<f32> {{
    let i_z = vec2<f32>(-z.y, z.x);
    let one = vec2<f32>(1.0, 0.0);
    let inside_sqrt = sub(one, mul(z, z));
    let root = c_sqrt(inside_sqrt);
    let sum = add(i_z, root);
    let ln = c_log(sum);
    return vec2<f32>(ln.y, -ln.x);
}}
fn c_acos(z: vec2<f32>) -> vec2<f32> {{
    let a = c_asin(z);
    return vec2<f32>(PI * 0.5 - a.x, -a.y);
}}
fn c_atan(z: vec2<f32>) -> vec2<f32> {{
    let i = vec2<f32>(0.0, 1.0);
    let num = add(i, z);
    let den = sub(i, z);
    let frac = div(num, den);
    let ln = c_log(frac);
    return vec2<f32>(-0.5 * ln.y, 0.5 * ln.x);
}}
fn c_asinh(z: vec2<f32>) -> vec2<f32> {{
    let one = vec2<f32>(1.0, 0.0);
    let inside = add(mul(z, z), one);
    let root = c_sqrt(inside);
    return c_log(add(z, root));
}}
fn c_acosh(z: vec2<f32>) -> vec2<f32> {{
    let one = vec2<f32>(1.0, 0.0);
    let zp = add(z, one);
    let zm = sub(z, one);
    let root = mul(c_sqrt(zp), c_sqrt(zm));
    return c_log(add(z, root));
}}
fn c_atanh(z: vec2<f32>) -> vec2<f32> {{
    let one = vec2<f32>(1.0, 0.0);
    let num = add(one, z);
    let den = sub(one, z);
    let frac = div(num, den);
    let ln = c_log(frac);
    return vec2<f32>(0.5 * ln.x, 0.5 * ln.y);
}}
fn c_log10(z: vec2<f32>) -> vec2<f32> {{
    let natural = c_log(z);
    let inv_ln10 = 1.0 / log(10.0);
    return vec2<f32>(natural.x * inv_ln10, natural.y * inv_ln10);
}}

fn user_function(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {{ return {expr}; }}

fn coord_to_pixel(z: vec2<f32>) -> vec2<i32> {{
    let fx = (z.x - params.cx0) / (params.cx1 - params.cx0);
    let fy = (z.y - params.cy0) / (params.cy1 - params.cy0);
    let px = i32(fx * f32(params.width));
    let py = i32(fy * f32(params.height));
    return vec2<i32>(px, py);
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let point_idx = gid.x;
    if (point_idx >= params.num_points) {{ return; }}
    
    let c = points[point_idx];
    var z = vec2<f32>(0.0, 0.0);
    
    // First pass: check if orbit escapes and count valid points
    var test_z = vec2<f32>(0.0, 0.0);
    var i: u32 = 0u;
    var escaped = false;
    
    loop {{
        if (i >= params.max_iters) {{ break; }}
        test_z = user_function(test_z, c);
        i = i + 1u;
        if (escape_abs2(test_z) > ESCAPE) {{
            escaped = true;
            break;
        }}
    }}
    
    // Decide whether to trace this orbit
    let should_trace = {condition};
    if (!should_trace) {{ return; }}
    
    // Second pass: trace and accumulate
    z = vec2<f32>(0.0, 0.0);
    for (var j: u32 = 0u; j < i; j = j + 1u) {{
        z = user_function(z, c);
        
        // Skip first iteration to avoid grid lines
        if (j > 0u) {{
            let pix = coord_to_pixel(z);
            if (pix.x >= 0 && pix.x < i32(params.width) && 
                pix.y >= 0 && pix.y < i32(params.height)) {{
                let idx = u32(pix.x) * params.height + u32(pix.y);
                atomicAdd(&bitmap[idx], 1u);
            }}
        }}
    }}
}}
"#,
        escape = max_abs as f32,
        expr = expr_wgsl,
        traj_size = traj_size,
        condition = if is_antibuddhabrot {
            "!escaped" // For antibuddhabrot, keep non-escaping orbits
        } else {
            "escaped" // For buddhabrot, keep escaping orbits
        }
    )
}

/// Generate sampling points for Buddhabrot on GPU
fn generate_sample_points(
    fractal: &Fractal,
    mandelbrot_bitmap: &Bitmap,
    rounds: u32,
    is_antibuddhabrot: bool,
) -> Vec<Complex<f32>> {
    let mut points = Vec::new();

    let top_right = Complex::new(
        x_to_coord(
            0,
            fractal.width,
            fractal.height,
            fractal.shift.re,
            fractal.zoom,
        ),
        y_to_coord(
            0,
            fractal.width,
            fractal.height,
            fractal.shift.im,
            fractal.zoom,
        ),
    );
    let bottom_left = Complex::new(
        x_to_coord(
            fractal.width,
            fractal.width,
            fractal.height,
            fractal.shift.re,
            fractal.zoom,
        ),
        y_to_coord(
            fractal.height,
            fractal.width,
            fractal.height,
            fractal.shift.im,
            fractal.zoom,
        ),
    );

    let samples_per_axis = (rounds as f64).sqrt() as i32;
    let x_step = (bottom_left.re - top_right.re) / samples_per_axis as f64;
    let y_step = (bottom_left.im - top_right.im) / samples_per_axis as f64;

    for i in 0..samples_per_axis {
        for j in 0..samples_per_axis {
            let x = top_right.re + i as f64 * x_step;
            let y = top_right.im + j as f64 * y_step;

            // Convert to pixel coordinates for Mandelbrot lookup
            let px = ((x - top_right.re) / (bottom_left.re - top_right.re) * fractal.width as f64)
                as i32;
            let py = ((y - top_right.im) / (bottom_left.im - top_right.im) * fractal.height as f64)
                as i32;

            if px >= 0 && px < fractal.width && py >= 0 && py < fractal.height {
                let mandel_val = mandelbrot_bitmap[(px * fractal.height + py) as usize];

                // Filter based on Mandelbrot set membership
                let should_include = if is_antibuddhabrot {
                    mandel_val >= fractal.iterations
                } else {
                    mandel_val < fractal.iterations
                };

                if should_include {
                    points.push(Complex::new(x as f32, y as f32));
                }
            }
        }
    }

    points
}

/// GPU-accelerated Buddhabrot renderer
pub fn render_buddhabrot_gpu<'a>(
    fractal: &Fractal,
    formula: &str,
    rounds: u32,
    is_antibuddhabrot: bool,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
) -> Bitmap {
    reset_cache();

    let start = Instant::now();

    // First, generate Mandelbrot set for filtering
    println!("Generating Mandelbrot pre-filter...");
    let mandelbrot_bitmap = {
        use super::gpu_engine::render_mandelbrot;
        render_mandelbrot(fractal, formula, device, queue)
    };

    // Generate sample points
    println!("Generating sample points...");
    let points = generate_sample_points(fractal, &mandelbrot_bitmap, rounds, is_antibuddhabrot);
    println!("Generated {} sample points", points.len());

    if points.is_empty() {
        println!("No points to process, returning empty bitmap");
        return [0u32; crate::config::MAX_PIXELS as usize];
    }

    // Parse formula and generate shader
    let ast = Parser::parse(formula);
    let expr_wgsl = gen_wgsl_expr(&ast);
    let shader_src = build_buddhabrot_shader(
        &expr_wgsl,
        is_antibuddhabrot,
        fractal.max_abs,
        fractal.iterations,
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("buddhabrot_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // Create bind group layout
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("buddhabrot_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        label: Some("buddhabrot_pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("buddhabrot_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    // Compute coordinates
    let cx0 = x_to_coord(
        0,
        fractal.width,
        fractal.height,
        fractal.shift.re,
        fractal.zoom,
    ) as f32;
    let cy0 = y_to_coord(
        0,
        fractal.width,
        fractal.height,
        fractal.shift.im,
        fractal.zoom,
    ) as f32;
    let cx1 = x_to_coord(
        fractal.width,
        fractal.width,
        fractal.height,
        fractal.shift.re,
        fractal.zoom,
    ) as f32;
    let cy1 = y_to_coord(
        fractal.height,
        fractal.width,
        fractal.height,
        fractal.shift.im,
        fractal.zoom,
    ) as f32;

    // Process in smaller batches to avoid timeouts
    const BATCH_SIZE: usize = 100_000;
    let num_batches = (points.len() + BATCH_SIZE - 1) / BATCH_SIZE;

    // Create output bitmap buffer (persistent across batches)
    let bitmap_size = (fractal.width as u64 * fractal.height as u64 * 4) as u64;
    let bitmap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bitmap"),
        size: bitmap_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bitmap_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    println!(
        "Processing {} batches of ~{} points each...",
        num_batches, BATCH_SIZE
    );

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * BATCH_SIZE;
        let batch_end = ((batch_idx + 1) * BATCH_SIZE).min(points.len());
        let batch_points = &points[batch_start..batch_end];

        if batch_idx % 10 == 0 || batch_idx == num_batches - 1 {
            println!(
                "Processing batch {}/{} ({} points)",
                batch_idx + 1,
                num_batches,
                batch_points.len()
            );
        }

        // Create params
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            width: u32,
            height: u32,
            max_iters: u32,
            num_points: u32,
            cx0: f32,
            cy0: f32,
            cx1: f32,
            cy1: f32,
        }
        unsafe impl bytemuck::Pod for Params {}
        unsafe impl bytemuck::Zeroable for Params {}

        let params = Params {
            width: fractal.width as u32,
            height: fractal.height as u32,
            max_iters: fractal.iterations,
            num_points: batch_points.len() as u32,
            cx0,
            cy0,
            cx1,
            cy1,
        };

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create points buffer
        let points_data: Vec<[f32; 2]> = batch_points.iter().map(|c| [c.re, c.im]).collect();
        let points_bytes = bytemuck::cast_slice(&points_data);
        let points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points"),
            size: points_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&points_buffer, 0, points_bytes);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buddhabrot_bind_group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bitmap_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buddhabrot_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("buddhabrot_pass"),
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (batch_points.len() as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Read back results
    println!("Reading back results...");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(&bitmap_buffer, 0, &readback_buffer, 0, bitmap_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    block_on(receiver.receive()).unwrap().unwrap();

    let data = slice.get_mapped_range().to_vec();
    readback_buffer.unmap();

    // Convert to bitmap
    let mut result = [0u32; crate::config::MAX_PIXELS as usize];
    for i in 0..(fractal.width as usize * fractal.height as usize) {
        let off = i * 4;
        result[i] = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
    }

    println!("Total time: {:.2?}", start.elapsed());
    result
}
