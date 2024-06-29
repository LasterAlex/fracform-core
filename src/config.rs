pub const MAX_PIXELS: u32 = 2000 * 2000;
pub static mut JOBS: u32 = 16; // For the future its mut
pub const STACK_SIZE: usize = 512 * 1024 * 1024 as usize;
pub const GENERATED_DIR: &str = "generated";
pub const ANIMATIONS_DIR: &str = "animations";
pub const FRACTALS_DIR: &str = "fractals";
pub const WRITE_TO_BITMAP_LEN_THRESHOLD: usize = 100000000; // Otherwise it will blow the fuck up
