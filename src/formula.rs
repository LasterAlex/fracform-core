use libloading::Library;
use num::Complex;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::Path;
use std::process::Command;

pub fn create_formula_project(user_function: &str) -> std::io::Result<bool> {
    // The bool is for "is it already present"
    let project_dir = Path::new("formula_project");
    let src_dir = project_dir.join("src");

    let main_rs_content = format!(
        r#"use num::Complex;

#[no_mangle]
pub extern "C" fn user_function(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {{
    {}
}}
"#,
        user_function
    );
    // If the lib.rs already has the same code, we are done
    if fs::read_to_string(src_dir.join("lib.rs")).unwrap_or("".to_string()) == main_rs_content {
        return Ok(false);
    }

    // Create directories
    fs::create_dir_all(&src_dir)?;

    // Generate Cargo.toml
    let cargo_toml_content = r#"[package]
name = "formula_project"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
num = "0.4""#;
    let mut cargo_toml = File::create(project_dir.join("Cargo.toml"))?;
    cargo_toml.write_all(cargo_toml_content.as_bytes())?;

    // Generate lib.rs with the user function
    let mut main_rs = File::create(src_dir.join("lib.rs"))?;
    main_rs.write_all(main_rs_content.as_bytes())?;

    Ok(true)
}

pub fn compile_formula_project() -> std::io::Result<()> {
    let output = Command::new("cargo")
        .env("RUSTFLAGS", "-Ctarget-cpu=native") // It will always be compiled, so, like, obv
        .args(&["build", "--release"])
        .current_dir("formula_project")
        .output()?;

    if !output.status.success() {
        eprintln!(
            "Failed to compile: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(target_os = "linux")]
const LIB_PATH: &str = "formula_project/target/release/libformula_project.so";
#[cfg(target_os = "windows")]
const LIB_PATH: &str = "formula_project/target/release/formula_project.dll";
#[cfg(target_os = "macos")]
const LIB_PATH: &str = "formula_project/target/release/libformula_project.dylib";

static mut LIB: Option<Library> = None;
static mut FUNC: Option<unsafe extern "C" fn(Complex<f64>, Complex<f64>) -> Complex<f64>> = None;

pub fn load_library() {
    unsafe {
        // Otherwise when the animations start, the library will be loaded one, and then it will
        // reuse the same file, not updating it
        if let Some(lib) = LIB.take() {
            lib.close().unwrap();
        }
        LIB = Some(Library::new(LIB_PATH).expect("Failed to load library"));
        // Just some fuckery to get the function we need
        let func: unsafe extern "C" fn(Complex<f64>, Complex<f64>) -> Complex<f64> = *LIB
            .as_ref()
            .unwrap()
            .get::<unsafe extern "C" fn(Complex<f64>, Complex<f64>) -> Complex<f64>>(
                b"user_function",
            )
            .expect("Failed to find function in library")
            .clone();
        FUNC = Some(func);
    }
}

pub fn execute_function(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
    unsafe { FUNC.unwrap()(z, c) }
}
