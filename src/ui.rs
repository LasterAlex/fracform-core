use eframe::egui;
use num::Complex;
use std::sync::mpsc;
use std::thread;

use crate::config::STACK_SIZE;
use crate::formula::{compile_formula_project, create_formula_project, load_library};
use crate::fractals::{x_to_coord, y_to_coord};
use crate::random_formula::get_random_formula;
use crate::{
    colors::PaletteMode,
    fractals::{Fractal, FractalType},
    make_fractal,
};

pub fn run_app() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Fracform",
        options,
        Box::new(|_cc| Ok(Box::new(FracformApp::new()))),
    )
    .unwrap();
}

#[derive(PartialEq, Clone)]
enum ImageState {
    NoImage,
    Generating,
    Done,
}

struct FracformApp {
    // Main fractal parameters
    params: FractalParams,
    fractal_type: FractalType,
    palette: PaletteMode,
    formula: String,

    // Main fractal state
    fractal_image: Option<Vec<Vec<(u8, u8, u8)>>>,
    fractal_texture: Option<egui::TextureHandle>,
    image_state: ImageState,

    // Mini Julia
    mini_julia: MiniJuliaData,

    // Thread communication
    fractal_thread: Option<thread::JoinHandle<()>>,
    fractal_receiver: Option<mpsc::Receiver<Vec<Vec<(u8, u8, u8)>>>>,
    julia_thread: Option<thread::JoinHandle<()>>,
    julia_receiver: Option<mpsc::Receiver<Vec<Vec<(u8, u8, u8)>>>>,

    // Selection fields
    selecting: bool,
    select_start: egui::Pos2,
    select_end: egui::Pos2,
}

#[derive(Debug)]
struct FractalParams {
    jobs: i32,
    width: i32,
    height: i32,
    zoom: f64,
    cx: f64,
    cy: f64,
    movex: f64,
    movey: f64,
    iterations: u32,
    abs_z: u32,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            jobs: 16,
            width: 1000,
            height: 1000,
            zoom: 0.5,
            cx: 0.0,
            cy: 0.0,
            movex: 0.0,
            movey: 0.0,
            iterations: 100,
            abs_z: 32,
        }
    }
}

struct MiniJuliaData {
    params: FractalParams,
    image_state: ImageState,
    fractal_image: Option<Vec<Vec<(u8, u8, u8)>>>,
    fractal_texture: Option<egui::TextureHandle>,
}

impl Default for MiniJuliaData {
    fn default() -> Self {
        let mut params = FractalParams::default();
        params.width = 500;
        params.height = 500;
        Self {
            params,
            image_state: ImageState::NoImage,
            fractal_image: None,
            fractal_texture: None,
        }
    }
}

impl FracformApp {
    fn new() -> Self {
        let mut params = FractalParams::default();
        params.width = 1000;
        params.height = 1000;
        Self {
            params,
            fractal_type: FractalType::Mandelbrot,
            palette: PaletteMode::Rainbow { offset: None },
            formula: "z * z + c".to_string(),
            fractal_image: None,
            fractal_texture: None,
            image_state: ImageState::NoImage,
            mini_julia: MiniJuliaData::default(),
            fractal_thread: None,
            fractal_receiver: None,
            julia_thread: None,
            julia_receiver: None,
            selecting: false,
            select_start: egui::Pos2::ZERO,
            select_end: egui::Pos2::ZERO,
        }
    }

    fn check_thread_completion(&mut self, ctx: &egui::Context) {
        // Check main fractal thread
        if let Some(receiver) = &self.fractal_receiver {
            if let Ok(image) = receiver.try_recv() {
                self.fractal_image = Some(image);
                self.image_state = ImageState::Done;
                self.fractal_thread = None;
                self.fractal_receiver = None;

                // Extract values first
                let image_data = self.fractal_image.as_ref();
                let width = self.params.width;
                let height = self.params.height;

                Self::update_texture(
                    ctx,
                    image_data,
                    width,
                    height,
                    &mut self.fractal_texture,
                    "fractal_texture",
                );
            } else {
                // Request repaint to keep checking while generating
                ctx.request_repaint();
            }
        }

        // Check mini julia thread
        if let Some(receiver) = &self.julia_receiver {
            if let Ok(image) = receiver.try_recv() {
                self.mini_julia.fractal_image = Some(image);
                self.mini_julia.image_state = ImageState::Done;
                self.julia_thread = None;
                self.julia_receiver = None;

                // Extract values first to avoid multiple borrows
                let image_data = self.mini_julia.fractal_image.as_ref();
                let width = self.mini_julia.params.width;
                let height = self.mini_julia.params.height;

                // Now update texture with extracted values
                Self::update_texture(
                    ctx,
                    image_data,
                    width,
                    height,
                    &mut self.mini_julia.fractal_texture,
                    "julia_texture",
                );
            } else {
                // Request repaint to keep checking while generating
                ctx.request_repaint();
            }
        }
    }

    fn update_texture(
        ctx: &egui::Context,
        image: Option<&Vec<Vec<(u8, u8, u8)>>>,
        width: i32,
        height: i32,
        texture: &mut Option<egui::TextureHandle>,
        name: &str,
    ) {
        if let Some(image_data) = image {
            let width = width as usize;
            let height = height as usize;

            let mut pixels = Vec::with_capacity(width * height * 4);
            for y in 0..height {
                for x in 0..width {
                    if y < image_data.len() && x < image_data[y].len() {
                        let (r, g, b) = image_data[x][y];
                        pixels.push(r);
                        pixels.push(g);
                        pixels.push(b);
                        pixels.push(255);
                    } else {
                        pixels.extend_from_slice(&[0, 0, 0, 255]);
                    }
                }
            }

            let color_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels);
            *texture = Some(ctx.load_texture(name, color_image, egui::TextureOptions::default()));
        }
    }

    fn start_fractal_generation(&mut self) {
        if self.fractal_thread.is_some() {
            return
        }

        let (tx, rx) = mpsc::channel();

        let params = self.params.clone();
        let fractal_type = self.fractal_type.clone();
        let palette = self.palette.clone();
        let formula = self.formula.clone();

        self.compile_formula_if_needed();

        let child = thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(move || {
                let mut fractal = Fractal::new(
                    params.width,
                    params.height,
                    params.zoom,
                    Complex::new(params.movex, params.movey),
                    params.iterations,
                    params.abs_z,
                    Some(Complex::new(params.cx, params.cy)),
                    palette,
                );
                let bitmap = make_fractal(&mut fractal, fractal_type, &formula);
                let _ = tx.send(bitmap);
            })
            .unwrap();

        self.fractal_thread = Some(child);
        self.fractal_receiver = Some(rx);
        self.image_state = ImageState::Generating;
        self.fractal_texture = None;
    }

    fn start_mini_julia_generation(&mut self) {
        if self.julia_thread.is_some() {
            return
        }

        let (tx, rx) = mpsc::channel();

        let params = self.mini_julia.params.clone();
        let palette = self.palette.clone();
        let formula = self.formula.clone();

        self.compile_formula_if_needed();

        let child = thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(move || {
                let mut fractal = Fractal::new(
                    params.width,
                    params.height,
                    params.zoom,
                    Complex::new(params.movex, params.movey),
                    params.iterations,
                    params.abs_z,
                    Some(Complex::new(params.cx, params.cy)),
                    palette,
                );
                let bitmap = make_fractal(&mut fractal, FractalType::Julia, &formula);
                let _ = tx.send(bitmap);
            })
            .unwrap();

        self.julia_thread = Some(child);
        self.julia_receiver = Some(rx);
        self.mini_julia.image_state = ImageState::Generating;
        self.mini_julia.fractal_texture = None;
    }

    fn compile_formula_if_needed(&self) {
        if create_formula_project(&self.formula).expect("Failed to generate C code") {
            compile_formula_project().expect("Failed to compile C code");
        }
        load_library();
    }
}

impl Clone for FractalParams {
    fn clone(&self) -> Self {
        Self {
            jobs: self.jobs,
            width: self.width,
            height: self.height,
            zoom: self.zoom,
            cx: self.cx,
            cy: self.cy,
            movex: self.movex,
            movey: self.movey,
            iterations: self.iterations,
            abs_z: self.abs_z,
        }
    }
}

impl eframe::App for FracformApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_thread_completion(ctx);

        // Always show mini julia as separate window when not in Julia mode
        if !matches!(self.fractal_type, FractalType::Julia) {
            self.show_mini_julia_window(ctx);
        }

        self.show_main_interface(ctx);
    }
}

// UI Components
impl FracformApp {
    fn show_mini_julia_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("Mini Julia Preview")
            .default_pos([ctx.screen_rect().width() - 320.0, 20.0])
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    // Image preview
                    Self::show_image_preview(
                        ui,
                        &self.mini_julia.image_state,
                        &self.mini_julia.fractal_texture,
                        300.0,
                    );

                    ui.add_space(4.0);
                    ui.set_width(300.0);

                    Self::show_fractal_params_ui(ui, &mut self.mini_julia.params, 600);

                    ui.add_space(4.0);

                    // Fix: Extract the state to avoid borrowing issues
                    let julia_state = self.mini_julia.image_state.clone();
                    let response =
                        Self::show_generate_button(ui, &julia_state, "Generate Mini Julia");
                    if response.clicked() {
                        self.start_mini_julia_generation();
                    }
                });
            });
    }

    fn show_main_interface(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Left panel - Parameters
                self.show_parameters_panel(ui);

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // Right panel - Image Display
                self.show_image_panel(ui);
            });
        });
    }

    fn show_parameters_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.set_width(280.0);

                    ui.heading("Fracform");
                    ui.add_space(8.0);

                    Self::show_collapsible_section(ui, "Tech Parameters", false, |ui| {
                        aligned_numeric_field(ui, "Jobs", &mut self.params.jobs);
                    });

                    Self::show_collapsible_section(ui, "Image Dimensions", false, |ui| {
                        aligned_numeric_field(ui, "Width", &mut self.params.width);
                        aligned_numeric_field(ui, "Height", &mut self.params.height);
                        self.params.width = self.params.width.clamp(1, 10000);
                        self.params.height = self.params.height.clamp(1, 10000);
                    });

                    self.show_fractal_type_ui(ui);
                    self.show_palette_ui(ui);

                    Self::show_collapsible_section(ui, "Position & Zoom", false, |ui| {
                        aligned_float_field(ui, "Zoom", &mut self.params.zoom);
                        aligned_double_float_field(
                            ui,
                            "Cx, Cy",
                            &mut self.params.cx,
                            &mut self.params.cy,
                        );
                        aligned_double_float_field(
                            ui,
                            "Move X, Y",
                            &mut self.params.movex,
                            &mut self.params.movey,
                        );

                        if ui.button("Reset zoom").clicked() {
                            let defaults = FractalParams::default();
                            self.params.zoom = defaults.zoom;
                            self.params.movex = defaults.movex;
                            self.params.movey = defaults.movey;

                            self.start_fractal_generation();
                        }
                    });

                    Self::show_collapsible_section(ui, "Fractal Parameters", false, |ui| {
                        aligned_numeric_field_u32(ui, "Iterations", &mut self.params.iterations);
                        aligned_numeric_field_u32(ui, "Abs Z", &mut self.params.abs_z);
                    });

                    Self::show_collapsible_section(ui, "Formula", true, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Formula:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.formula).desired_width(150.0),
                            );
                        });
                        if ui.button("Random formula").clicked() {
                            self.formula = get_random_formula();
                            self.start_fractal_generation();
                        }
                    });

                    ui.add_space(12.0);

                    // Fix: Extract the state to avoid borrowing issues
                    let main_state = self.image_state.clone();
                    let response = Self::show_generate_button(ui, &main_state, "Generate Fractal");
                    if response.clicked() {
                        self.start_fractal_generation();
                    }
                });
            });
    }

    fn show_image_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.heading("Preview");
                    ui.add_space(8.0);

                    if let Some(double_click) = self.show_main_image_preview(
                        ui,
                        self.image_state.clone(),
                        self.fractal_texture.clone(),
                        self.params.width,
                        self.params.height,
                    ) {
                        let x_coord = x_to_coord(
                            double_click.x as i32,
                            self.params.width,
                            self.params.height,
                            self.params.movex,
                            self.params.zoom,
                        );
                        let y_coord = y_to_coord(
                            double_click.y as i32,
                            self.params.width,
                            self.params.height,
                            self.params.movey,
                            self.params.zoom,
                        );
                        self.mini_julia.params.cx = x_coord;
                        self.mini_julia.params.cy = y_coord;
                        self.mini_julia.params.zoom = 0.5;
                        self.mini_julia.params.movex = 0.0;
                        self.mini_julia.params.movey = 0.0;
                        self.start_mini_julia_generation();
                    }
                });
            });
    }

    fn show_image_preview(
        ui: &mut egui::Ui,
        state: &ImageState,
        texture: &Option<egui::TextureHandle>,
        size: f32,
    ) {
        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            ui.set_min_size(egui::vec2(size, size));
            ui.set_max_size(egui::vec2(size, size));
            ui.centered_and_justified(|ui| match state {
                ImageState::Done => {
                    if let Some(texture) = texture {
                        ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(size, size)));
                    } else {
                        ui.label(egui::RichText::new("Preview Ready").color(egui::Color32::GREEN));
                    }
                }
                ImageState::Generating => {
                    ui.label(egui::RichText::new("Generating...").color(egui::Color32::YELLOW));
                }
                ImageState::NoImage => {
                    ui.label(
                        egui::RichText::new("Click to generate").color(egui::Color32::DARK_GRAY),
                    );
                }
            });
        });
    }

    fn show_main_image_preview(
        &mut self,
        ui: &mut egui::Ui,
        state: ImageState,
        texture: Option<egui::TextureHandle>,
        width: i32,
        height: i32,
    ) -> Option<egui::Vec2> {
        let mut double_click_pos = None;

        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            ui.set_min_size(egui::vec2(400.0, 400.0));
            ui.set_max_size(egui::vec2(600.0, 600.0));

            ui.centered_and_justified(|ui| match state {
                ImageState::Done => {
                    if let Some(texture) = texture {
                        let available = ui.available_size();
                        let image_size = egui::vec2(width as f32, height as f32);

                        let scale = available.x.min(available.y) / image_size.x.max(image_size.y);
                        let display_size = image_size * scale;

                        let (rect, response) =
                            ui.allocate_exact_size(display_size, egui::Sense::click_and_drag());

                        ui.painter().image(
                            texture.id(),
                            rect,
                            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );

                        // -------------------------
                        // Crosshair OR selection-center marker
                        // -------------------------

                        if self.selecting {
                            let center_pos = (self.select_start + self.select_end.to_vec2()) * 0.5;
                            let half = 3.0;
                            let marker_rect = egui::Rect::from_min_max(
                                egui::pos2(center_pos.x - half, center_pos.y - half),
                                egui::pos2(center_pos.x + half, center_pos.y + half),
                            );
                            ui.painter().rect_stroke(
                                marker_rect,
                                0.0,
                                egui::Stroke::new(1.0, egui::Color32::WHITE),
                            );
                        }

                        // -------------------------
                        // Mouse interaction
                        // -------------------------
                        if response.double_clicked() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let image_pos = (pos - rect.min) / scale;
                                double_click_pos = Some(image_pos);
                            }
                        }

                        if response.drag_started() {
                            if let Some(mouse) = response.interact_pointer_pos() {
                                self.selecting = true;
                                self.select_start = mouse;
                                self.select_end = mouse;
                            }
                        }

                        // if self.selecting && response.dragged() {
                        //     if let Some(mouse) = response.interact_pointer_pos() {
                        //         let aspect = self.params.width as f32 / self.params.height as f32;
                        //         self.select_end =
                        //             Self::enforce_aspect_ratio(self.select_start, mouse, aspect);
                        //     }
                        // }
                        if self.selecting && response.dragged() {
                            if let Some(mouse) = response.interact_pointer_pos() {
                                self.select_end = mouse;
                            }
                        }

                        if self.selecting && response.drag_stopped() {
                            self.selecting = false;

                            // Convert selection rect to image coordinates
                            let p1 = (self.select_start - rect.min) / scale;
                            let p2 = (self.select_end - rect.min) / scale;

                            let min = egui::Vec2::new(p1.x.min(p2.x), p1.y.min(p2.y));
                            let max = egui::Vec2::new(p1.x.max(p2.x), p1.y.max(p2.y));

                            self.start_zoom(min.x, min.y, max.x, max.y);
                        }

                        // -------------------------
                        // Draw selection rectangle
                        // -------------------------
                        if self.selecting {
                            let r = egui::Rect::from_two_pos(self.select_start, self.select_end);
                            ui.painter().rect_stroke(
                                r,
                                0.0,
                                egui::Stroke::new(1.0, egui::Color32::GREEN),
                            );
                        }
                    }
                }
                ImageState::Generating => {
                    ui.label("Generating fractal…");
                }
                ImageState::NoImage => {
                    ui.label("No image generated.");
                }
            });
        });

        double_click_pos
    }

    fn start_zoom(&mut self, top_x: f32, top_y: f32, bottom_x: f32, bottom_y: f32) {
        let height = (bottom_y - top_y).abs();
        let width = (bottom_x - top_x).abs();
        if height.min(width) < 30.0 {
            return;
        }

        let width = self.params.width;
        let height = self.params.height;
        let shift_x = self.params.movex;
        let shift_y = self.params.movey;
        let zoom = self.params.zoom;

        let top_x_coord = x_to_coord(top_x as i32, width, height, shift_x, zoom);
        let top_y_coord = y_to_coord(top_y as i32, width, height, shift_y, zoom);

        let bottom_x_coord = x_to_coord(bottom_x as i32, width, height, shift_x, zoom);
        let bottom_y_coord = y_to_coord(bottom_y as i32, width, height, shift_y, zoom);

        let new_shift_x = (bottom_x_coord + top_x_coord) / 2.0;
        let new_shift_y = (bottom_y_coord + top_y_coord) / 2.0;

        let image_width = (x_to_coord(width, width, height, shift_x, zoom)
            - x_to_coord(0, width, height, shift_x, zoom))
        .abs();

        let image_height = (y_to_coord(height, width, height, shift_y, zoom)
            - y_to_coord(0, width, height, shift_y, zoom))
        .abs();

        let new_width = (bottom_x_coord - top_x_coord).abs();
        let new_height = (bottom_y_coord - top_y_coord).abs();

        let new_zoom = if new_width > new_height {
            image_width / new_width * zoom
        } else {
            image_height / new_height * zoom
        };

        self.params.movex = new_shift_x;
        self.params.movey = new_shift_y;
        self.params.zoom = new_zoom;

        self.start_fractal_generation();
    }

    fn show_fractal_params_ui(ui: &mut egui::Ui, params: &mut FractalParams, max_size: i32) {
        Self::show_collapsible_section(ui, "Image Dimensions", false, |ui| {
            aligned_numeric_field(ui, "Width", &mut params.width);
            aligned_numeric_field(ui, "Height", &mut params.height);
            params.width = params.width.clamp(1, max_size);
            params.height = params.height.clamp(1, max_size);
        });

        Self::show_collapsible_section(ui, "Position & Zoom", false, |ui| {
            aligned_float_field(ui, "Zoom", &mut params.zoom);
            aligned_double_float_field(ui, "Cx, Cy", &mut params.cx, &mut params.cy);
            aligned_double_float_field(ui, "Move X, Y", &mut params.movex, &mut params.movey);
        });

        Self::show_collapsible_section(ui, "Fractal Parameters", false, |ui| {
            aligned_numeric_field_u32(ui, "Iterations", &mut params.iterations);
            aligned_numeric_field_u32(ui, "Abs Z", &mut params.abs_z);
        });
    }

    fn show_fractal_type_ui(&mut self, ui: &mut egui::Ui) {
        Self::show_collapsible_section(ui, "Fractal Type", false, |ui| {
            egui::ComboBox::from_label("Type")
                .selected_text(fractal_type_name(&self.fractal_type))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.fractal_type,
                        FractalType::Mandelbrot,
                        "Mandelbrot",
                    );
                    ui.selectable_value(&mut self.fractal_type, FractalType::Julia, "Julia");

                    if ui
                        .selectable_label(
                            matches!(self.fractal_type, FractalType::Buddhabrot { .. }),
                            "Buddhabrot",
                        )
                        .clicked()
                    {
                        self.fractal_type = FractalType::Buddhabrot {
                            rounds: 100_000_000,
                        };
                    }
                    if ui
                        .selectable_label(
                            matches!(self.fractal_type, FractalType::Antibuddhabrot { .. }),
                            "Antibuddhabrot",
                        )
                        .clicked()
                    {
                        self.fractal_type = FractalType::Antibuddhabrot { rounds: 20_000_000 };
                    }
                    if ui
                        .selectable_label(
                            matches!(self.fractal_type, FractalType::Nebulabrot { .. }),
                            "Nebulabrot",
                        )
                        .clicked()
                    {
                        self.fractal_type = FractalType::Nebulabrot {
                            rounds: 100_000_000,
                            red_iters: 5000,
                            green_iters: 500,
                            blue_iters: 50,
                            color_shift: None,
                            uniform_factor: Some(0.5),
                        };
                    }
                    if ui
                        .selectable_label(
                            matches!(self.fractal_type, FractalType::Antinebulabrot { .. }),
                            "Antinebulabrot",
                        )
                        .clicked()
                    {
                        self.fractal_type = FractalType::Antinebulabrot {
                            rounds: 20_000_000,
                            red_iters: 5000,
                            green_iters: 500,
                            blue_iters: 50,
                            color_shift: None,
                            uniform_factor: None,
                        };
                    }
                });

            match &mut self.fractal_type {
                FractalType::Buddhabrot { rounds } | FractalType::Antibuddhabrot { rounds } => {
                    aligned_numeric_field_u32(ui, "Rounds", rounds);
                }
                FractalType::Nebulabrot {
                    rounds,
                    red_iters,
                    green_iters,
                    blue_iters,
                    color_shift,
                    uniform_factor,
                }
                | FractalType::Antinebulabrot {
                    rounds,
                    red_iters,
                    green_iters,
                    blue_iters,
                    color_shift,
                    uniform_factor,
                } => {
                    aligned_numeric_field_u32(ui, "Rounds", rounds);
                    aligned_numeric_field_u32(ui, "Red Iters", red_iters);
                    aligned_numeric_field_u32(ui, "Green Iters", green_iters);
                    aligned_numeric_field_u32(ui, "Blue Iters", blue_iters);
                    aligned_optional_u32(ui, "Color Shift", color_shift);
                    aligned_optional_f64(ui, "Uniform Factor", uniform_factor);
                }
                _ => {}
            }
        });
    }

    fn show_palette_ui(&mut self, ui: &mut egui::Ui) {
        Self::show_collapsible_section(ui, "Palette Mode", false, |ui| {
            egui::ComboBox::from_label("Palette")
                .selected_text(palette_mode_name(&self.palette))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.palette,
                        PaletteMode::BrownAndBlue,
                        "Brown and Blue",
                    );
                    ui.selectable_value(
                        &mut self.palette,
                        PaletteMode::BlackAndWhite,
                        "Black and White",
                    );
                    ui.selectable_value(&mut self.palette, PaletteMode::Custom, "Custom");

                    if ui
                        .selectable_label(
                            matches!(self.palette, PaletteMode::Rainbow { .. }),
                            "Rainbow",
                        )
                        .clicked()
                    {
                        self.palette = PaletteMode::Rainbow { offset: None };
                    }
                    if ui
                        .selectable_label(
                            matches!(self.palette, PaletteMode::Smooth { .. }),
                            "Smooth",
                        )
                        .clicked()
                    {
                        self.palette = PaletteMode::Smooth {
                            shift: None,
                            offset: None,
                        };
                    }
                    if ui
                        .selectable_label(
                            matches!(self.palette, PaletteMode::Naive { .. }),
                            "Naive",
                        )
                        .clicked()
                    {
                        self.palette = PaletteMode::Naive {
                            shift: None,
                            offset: None,
                        };
                    }
                    if ui
                        .selectable_label(
                            matches!(self.palette, PaletteMode::GrayScale { .. }),
                            "GrayScale",
                        )
                        .clicked()
                    {
                        self.palette = PaletteMode::GrayScale {
                            shift: None,
                            uniform_factor: Some(0.5),
                        };
                    }
                });

            match &mut self.palette {
                PaletteMode::Rainbow { offset } => {
                    aligned_optional_u32(ui, "Offset", offset);
                }
                PaletteMode::Smooth { shift, offset } | PaletteMode::Naive { shift, offset } => {
                    aligned_optional_u32(ui, "Shift", shift);
                    aligned_optional_u32(ui, "Offset", offset);
                }
                PaletteMode::GrayScale {
                    shift,
                    uniform_factor,
                } => {
                    aligned_optional_u32(ui, "Shift", shift);
                    aligned_optional_f64(ui, "Uniform Factor", uniform_factor);
                }
                _ => {}
            }
        });
    }

    fn show_collapsible_section(
        ui: &mut egui::Ui,
        title: &str,
        default_open: bool,
        content: impl FnOnce(&mut egui::Ui),
    ) {
        egui::CollapsingHeader::new(title)
            .default_open(default_open)
            .show(ui, |ui| {
                ui.set_width(260.0);
                content(ui);
            });
        ui.add_space(4.0);
    }

    fn show_generate_button(ui: &mut egui::Ui, state: &ImageState, text: &str) -> egui::Response {
        let enabled = *state != ImageState::Generating;
        let button_text = if *state == ImageState::Generating {
            "Generating..."
        } else {
            text
        };

        let button = egui::Button::new(button_text).min_size(egui::vec2(260.0, 32.0));
        ui.add_enabled(enabled, button)
    }
}

// Helper functions
fn fractal_type_name(fractal: &FractalType) -> &str {
    match fractal {
        FractalType::Mandelbrot => "Mandelbrot",
        FractalType::Julia => "Julia",
        FractalType::Buddhabrot { .. } => "Buddhabrot",
        FractalType::Antibuddhabrot { .. } => "Antibuddhabrot",
        FractalType::Nebulabrot { .. } => "Nebulabrot",
        FractalType::Antinebulabrot { .. } => "Antinebulabrot",
    }
}

fn palette_mode_name(palette: &PaletteMode) -> &str {
    match palette {
        PaletteMode::BrownAndBlue => "Brown and Blue",
        PaletteMode::BlackAndWhite => "Black and White",
        PaletteMode::Rainbow { .. } => "Rainbow",
        PaletteMode::Smooth { .. } => "Smooth",
        PaletteMode::Naive { .. } => "Naive",
        PaletteMode::Custom => "Custom",
        PaletteMode::GrayScale { .. } => "GrayScale",
    }
}

fn aligned_numeric_field(ui: &mut egui::Ui, label: &str, value: &mut i32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let id = ui.id().with(label);
            let mut buf = ui.data_mut(|d| {
                d.get_temp::<String>(id)
                    .unwrap_or_else(|| value.to_string())
            });

            let response = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(80.0));

            if let Ok(v) = buf.parse::<i32>() {
                *value = v;
            }

            // Store buffer if focused, otherwise reset to actual value
            if response.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, buf));
            } else {
                ui.data_mut(|d| d.remove::<String>(id));
            }
        });
    });
}

fn aligned_numeric_field_u32(ui: &mut egui::Ui, label: &str, value: &mut u32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let id = ui.id().with(label);
            let mut buf = ui.data_mut(|d| {
                d.get_temp::<String>(id)
                    .unwrap_or_else(|| value.to_string())
            });

            let response = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(80.0));

            if let Ok(v) = buf.parse::<u32>() {
                *value = v;
            }

            if response.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, buf));
            } else {
                ui.data_mut(|d| d.remove::<String>(id));
            }
        });
    });
}

fn aligned_optional_u32(ui: &mut egui::Ui, label: &str, value: &mut Option<u32>) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let id = ui.id().with(label);
            let mut buf = ui.data_mut(|d| {
                d.get_temp::<String>(id)
                    .unwrap_or_else(|| value.map_or(String::new(), |v| v.to_string()))
            });

            let response = ui.add(
                egui::TextEdit::singleline(&mut buf)
                    .hint_text("None")
                    .desired_width(80.0),
            );

            if buf.is_empty() {
                *value = None;
            } else if let Ok(v) = buf.parse::<u32>() {
                *value = Some(v);
            }

            if response.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, buf));
            } else {
                ui.data_mut(|d| d.remove::<String>(id));
            }
        });
    });
}

fn aligned_optional_f64(ui: &mut egui::Ui, label: &str, value: &mut Option<f64>) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let id = ui.id().with(label);
            let mut buf = ui.data_mut(|d| {
                d.get_temp::<String>(id)
                    .unwrap_or_else(|| value.map_or(String::new(), |v| v.to_string()))
            });

            let response = ui.add(
                egui::TextEdit::singleline(&mut buf)
                    .hint_text("None")
                    .desired_width(80.0),
            );

            if buf.is_empty() {
                *value = None;
            } else if let Ok(v) = buf.parse::<f64>() {
                *value = Some(v);
            }

            if response.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, buf));
            } else {
                ui.data_mut(|d| d.remove::<String>(id));
            }
        });
    });
}

fn aligned_float_field(ui: &mut egui::Ui, label: &str, value: &mut f64) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let id = ui.id().with(label);
            let mut buf = ui.data_mut(|d| {
                d.get_temp::<String>(id)
                    .unwrap_or_else(|| value.to_string())
            });

            let response = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(80.0));

            if let Ok(v) = buf.parse::<f64>() {
                *value = v;
            }

            if response.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, buf));
            } else {
                ui.data_mut(|d| d.remove::<String>(id));
            }
        });
    });
}

fn aligned_double_float_field(ui: &mut egui::Ui, label: &str, value1: &mut f64, value2: &mut f64) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let id2 = ui.id().with(label).with("field2");
            let mut buf2 = ui.data_mut(|d| {
                d.get_temp::<String>(id2)
                    .unwrap_or_else(|| value2.to_string())
            });

            let response2 = ui.add(egui::TextEdit::singleline(&mut buf2).desired_width(60.0));

            if let Ok(v) = buf2.parse::<f64>() {
                *value2 = v;
            }

            if response2.has_focus() {
                ui.data_mut(|d| d.insert_temp(id2, buf2));
            } else {
                ui.data_mut(|d| d.remove::<String>(id2));
            }

            let id1 = ui.id().with(label).with("field1");
            let mut buf1 = ui.data_mut(|d| {
                d.get_temp::<String>(id1)
                    .unwrap_or_else(|| value1.to_string())
            });

            let response1 = ui.add(egui::TextEdit::singleline(&mut buf1).desired_width(60.0));

            if let Ok(v) = buf1.parse::<f64>() {
                *value1 = v;
            }

            if response1.has_focus() {
                ui.data_mut(|d| d.insert_temp(id1, buf1));
            } else {
                ui.data_mut(|d| d.remove::<String>(id1));
            }
        });
    });
}
