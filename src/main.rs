use show_image::{ImageView, ImageInfo};

mod approximator;
use approximator::{load_image, display_image, save_image, QuadTree, Image, DivisionPolicy};

const NAME: &str = "shrek";
const TOLERANCE: f32 = 5.5;
const MIN_BOX_LEN: u32 = 4;
const SHOW_QUADTREE: bool = true;


#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> { 
    
    let path = format!("inputs/{}.png", NAME);

    let image = load_image(&path)
        .expect("Failed to load image");

    let width = image.width;
    let height = image.height;

    let min_dim = width.min(height);

    let quadtree = QuadTree::new(&image, TOLERANCE, 32, DivisionPolicy::Adaptive(7), MIN_BOX_LEN);
    // let quadtree = QuadTree::new(&image, TOLERANCE, 32, DivisionPolicy::Regular, MIN_BOX_LEN);

    println!("Quadtree constructed with {} leaves", quadtree.leaf_count());
    

    let image_pixels_approximated = quadtree.approximate(&image, SHOW_QUADTREE);

    let image = ImageView::new(ImageInfo::rgb8(width, height), &image_pixels_approximated);

    display_image(image)?;
    save_image(Image::new(image_pixels_approximated, width, height), NAME);
    

    std::thread::sleep(std::time::Duration::from_secs(30));


    Ok(())
}
