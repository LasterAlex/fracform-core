// Ported almost one to one from python with help of chatgpt, it works but isn't the best
use image::GenericImageView;
use rand::Rng;
use std::ffi::OsString;
use std::path::Path;
use std::process::Command;
use std::{fs, io};

fn filename(deep: i32, file: Option<String>) -> String {
    let mut file = file.unwrap_or_else(|| String::from(""));
    if deep <= 0 {
        return format!("frac_{}_{}", file, "");
    }
    let mut rng = rand::rng();
    if rng.random_bool(0.5) {
        file.push(rng.sample(rand::distr::Alphanumeric) as char);
    } else {
        file.push(rng.random_range(0..=9).to_string().chars().next().unwrap());
    }
    filename(deep - 1, Some(file))
}

fn add_0(path: &Path, max_number_amount: usize) {
    println!("Adding zeros...");
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let filename = entry.file_name();
        let mut filename_vec: Vec<char> = filename.to_str().unwrap().chars().collect();
        for _ in 0..max_number_amount - digits_in_frame(filename) {
            filename_vec.insert(0, '0');
        }
        let new_filename: String = filename_vec.into_iter().collect();
        fs::rename(entry.path(), path.join(new_filename)).unwrap();
    }
}

fn make_one_last_name(path: &Path, name: &str) {
    println!("Making one name for every frame...");
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let filename = entry.file_name();
        let final_filename = format!(
            "{}_{}",
            filename
                .to_str()
                .unwrap()
                .split("_")
                .collect::<Vec<&str>>()
                .first()
                .unwrap(),
            name
        );
        fs::rename(entry.path(), path.join(final_filename + ".png")).unwrap();
    }
}

fn digits_in_frame(path: OsString) -> usize {
    path.to_str()
        .unwrap()
        .split("_")
        .collect::<Vec<&str>>()
        .first()
        .unwrap()
        .len()
}

fn digits_in_last_frame(path: &Path) -> usize {
    let entries: Vec<_> = fs::read_dir(path).unwrap().collect();
    let mut max = 0;
    for entry in entries.iter() {
        let entry = entry.as_ref().unwrap();
        let filename = entry.file_name();
        let digits = digits_in_frame(filename);
        if digits > max {
            max = digits;
        }
    }
    max
}

fn resolution_first_frame(path: &Path) -> (u32, u32) {
    let entries: Vec<_> = fs::read_dir(path).unwrap().collect();
    let first_entry = entries.first().unwrap().as_ref().unwrap().path();
    let img = image::open(first_entry).unwrap();
    img.dimensions()
}

fn last_symbols_of_img(path: &Path) -> String {
    let entries: Vec<_> = fs::read_dir(path).unwrap().collect();
    let first_entry = entries.first().unwrap().as_ref().unwrap().file_name();
    let binding = first_entry.clone();
    let filename_str = binding.to_str().unwrap();
    filename_str
        .chars()
        .skip(digits_in_frame(first_entry))
        .collect()
}

fn name_of_directory(path: &Path) -> String {
    path.file_name().unwrap().to_str().unwrap().to_string()
}

fn directory_below_this(path: &Path) -> String {
    path.parent().unwrap().to_str().unwrap().to_string()
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}

pub fn make_mp4(path: &Path, fps: u32) {
    println!("_tpm directory making...");
    let binding =
        directory_below_this(path).to_string() + &format!("/tmp_animation_{}", filename(5, None));
    let tmp_path = Path::new(&binding);
    copy_dir_all(path, tmp_path).unwrap();
    println!("_tpm directory made");
    make_one_last_name(tmp_path, &filename(5, None));
    add_0(tmp_path, digits_in_last_frame(path));

    let resolution = resolution_first_frame(path);
    let resolution_str = format!("{}x{}", resolution.0, resolution.1);
    let path_ffmpeg = format!(
        "{}/%0{}d{}",
        tmp_path.to_str().unwrap(),
        digits_in_last_frame(tmp_path),
        last_symbols_of_img(tmp_path)
    );
    let name_mp4 = format!(
        "{}/{}.mp4",
        directory_below_this(path),
        name_of_directory(path)
    );
    let command = format!(
        "ffmpeg -r {fps} -i \"{path_ffmpeg}\" -c:v libx264 -s:v {resolution_str} -r 30 -pix_fmt yuv420p \"{name_mp4}\""
    );
    println!("{command}");
    println!("Generating mp4...");
    Command::new("sh")
        .arg("-c")
        .arg(&command)
        .output()
        .expect("failed to execute process");

    fs::remove_dir_all(tmp_path).unwrap();
    println!("_tmp directory deleted");
    println!("mp4 Ready to watch!");
    println!("Directory of mp4: {name_mp4}");
}
