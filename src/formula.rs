use libloading::Library;
use num::Complex;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::Path;
use std::process::Command;

pub fn create_formula_project(user_function: &str) -> std::io::Result<bool> {
    let project_dir = Path::new("formula_c");

    let c_content = format!(
        r#"#include <complex.h>
#include <math.h>

double complex user_function(double complex z, double complex c) {{
    return {user_function};
}}
"#
    );

    if fs::read_to_string(project_dir.join("formula.c")).unwrap_or_default() == c_content {
        return Ok(false);
    }

    fs::create_dir_all(project_dir)?;
    let mut c_file = File::create(project_dir.join("formula.c"))?;
    c_file.write_all(c_content.as_bytes())?;

    Ok(true)
}

pub fn compile_formula_project() -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    let lib_name = "formula_c/libformula.so";
    #[cfg(target_os = "windows")]
    let lib_name = "formula_c/formula.dll";
    #[cfg(target_os = "macos")]
    let lib_name = "formula_c/libformula.dylib";

    let mut cmd = Command::new("gcc");
    cmd.args([
        "-O3",
        "-march=native",
        "-ffast-math",
        "-shared",
        "-fPIC",
        "formula_c/formula.c",
        "-o",
        lib_name,
        "-lm",
    ]);

    #[cfg(target_os = "windows")]
    cmd.args(&["-Wl,--export-all-symbols"]);

    let output = cmd.output()?;

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
const LIB_PATH: &str = "formula_c/libformula.so";
#[cfg(target_os = "windows")]
const LIB_PATH: &str = "formula_c/formula.dll";
#[cfg(target_os = "macos")]
const LIB_PATH: &str = "formula_c/libformula.dylib";

static mut LIB: Option<Library> = None;
static mut FUNC: Option<unsafe extern "C" fn(Complex<f64>, Complex<f64>) -> Complex<f64>> = None;

pub fn load_library() {
    unsafe {
        if let Some(lib) = LIB.take() {
            let _ = lib.close();
        }
        LIB = Some(Library::new(LIB_PATH).expect("Failed to load library"));

        // C uses different calling convention for complex numbers
        // We need to handle this carefully
        let func: unsafe extern "C" fn(Complex<f64>, Complex<f64>) -> Complex<f64> = *LIB
            .as_ref()
            .unwrap()
            .get(b"user_function")
            .expect("Failed to find function in library");
        FUNC = Some(func);
    }
}

pub fn execute_function(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
    unsafe { FUNC.unwrap()(z, c) }
}
