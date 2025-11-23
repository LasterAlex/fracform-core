use image::{DynamicImage, GenericImageView};
use num::pow::Pow;

pub fn precompute_img(img: DynamicImage) -> Result<Vec<Vec<bool>>, Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();

    let mut bitmap = vec![];

    for x in 0..width as usize {
        bitmap.push(vec![]);
        for y in 0..height as usize {
            let img_pixel = img.get_pixel(x as u32, y as u32);

            let ground_truth_is_black = is_black_pixel((img_pixel[0], img_pixel[1], img_pixel[2]));
            bitmap[x].push(ground_truth_is_black)
        }
    }

    Ok(bitmap)
}

/// Compares shadow similarity between an image bitmap and a file
/// Returns a score between 0.0 (completely different) and 1.0 (identical)
pub fn compare_shadows(
    bitmap: &Vec<Vec<(u8, u8, u8)>>,
    img: &Vec<Vec<bool>>,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Load the comparison image
    let width = img.len();
    let height = img[0].len();

    // Check dimensions match
    if bitmap.len() != width as usize || (bitmap.len() > 0 && bitmap[0].len() != height as usize) {
        return Err("Image dimensions don't match".into());
    }

    // Threshold for considering a pixel as "black" (shadow)
    // A pixel is considered black if all RGB components are below this value

    let mut true_black = 0;
    let mut incorrect_black = 0;
    let mut true_white = 0;
    let mut incorrect_white = 0;

    // Compare pixels - filename is ground truth
    for y in 0..height as usize {
        for x in 0..width as usize {
            let bitmap_pixel = bitmap[x][y];

            let predicted_is_black = is_black_pixel(bitmap_pixel);
            let ground_truth_is_black = img[x][y];

            if ground_truth_is_black {
                true_black += 1;
            } else {
                true_white += 1;
            }

            // Check if prediction matches ground truth
            if predicted_is_black && !ground_truth_is_black {
                incorrect_black += 1;
            } else if !predicted_is_black && ground_truth_is_black {
                incorrect_white += 1;
            }
        }
    }

    // Return accuracy score
    let accuracy = (((true_white - incorrect_black) as f64 / true_white as f64).pow(2)) // punish
    // white spaces
        * (((true_black - incorrect_white) as f64 / true_black as f64).pow(1));
    Ok(accuracy)
}

/// Helper function to determine if a pixel is considered black (shadow)
fn is_black_pixel(pixel: (u8, u8, u8)) -> bool {
    let avg = (pixel.0 as u32 + pixel.1 as u32 + pixel.2 as u32) / 3;
    avg < 128
}

/// Returns true if the bitmap is either all black or has no black pixels
/// (i.e., not useful for shadow comparison)
pub fn is_bitmap_uniform(bitmap: &Vec<Vec<(u8, u8, u8)>>) -> bool {
    if bitmap.is_empty() || bitmap[0].is_empty() {
        return true;
    }

    let mut has_black = false;
    let mut has_non_black = false;

    for row in bitmap {
        for &pixel in row {
            if is_black_pixel(pixel) {
                has_black = true;
            } else {
                has_non_black = true;
            }

            // Early exit if we found both types
            if has_black && has_non_black {
                return false;
            }
        }
    }

    // True if only black OR only non-black (uniform)
    true
}
