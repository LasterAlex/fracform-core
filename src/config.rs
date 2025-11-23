use std::sync::Mutex;

pub const MAX_PIXELS: u32 = 3000 * 3010;
pub const JOBS: Mutex<u32> = Mutex::new(16); // For the future its mut
pub const STACK_SIZE: usize = 1024 * 1024 * 1024_usize;
pub const GENERATED_DIR: &str = "generated";
pub const ANIMATIONS_DIR: &str = "animations";
pub const FRACTALS_DIR: &str = "fractals";
pub const WRITE_TO_BITMAP_LEN_THRESHOLD: usize = 10000000; // Otherwise it will blow the fuck up
