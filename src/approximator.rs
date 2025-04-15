use show_image::{create_window, ImageView};

#[derive(Clone, Debug)]
struct Vec2D<T> {
    x: T,
    y: T,
}
impl<T> Vec2D<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T> From<Vec2D<T>> for (T, T) {
    fn from(vec: Vec2D<T>) -> Self {
        (vec.x, vec.y)
    }
}


pub struct Image {
    pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        Self { pixels, width, height }
    }

    fn get_pixel(&self, x: u32, y: u32) -> &[u8] {
        let index = (y * self.width + x) * 3;
        &self.pixels[index as usize..index as usize + 3]
    }

    fn get_pixel_from_coord(&self, coord: Vec2D<u32>) -> &[u8] {
        let index = (coord.y * self.width + coord.x) * 3;
        &self.pixels[index as usize..index as usize + 3]
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: &[u8]) {
        let index = (y * self.width + x) * 3;
        self.pixels[index as usize..index as usize + 3].copy_from_slice(color);
    }

    fn set_pixel_from_coord(&mut self, coord: Vec2D<u32>, color: &[u8]) {
        let index = (coord.y * self.width + coord.x) * 3;
        self.pixels[index as usize..index as usize + 3].copy_from_slice(color);
    }
}


pub fn load_image(path: &str) -> Result<Image, Box<dyn std::error::Error>> {
    let img = image::open(path)
        .expect("Failed to open image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            pixels.extend_from_slice(&[pixel[0], pixel[1], pixel[2]]);
        }
    }
    Ok(Image::new(pixels, width, height))
}

pub fn display_image(image: ImageView) -> Result<(), Box<dyn std::error::Error>> { 
    let window = create_window("approximation_visualizer", Default::default())?;
    window.set_image("image_display", image)?;
    Ok(())
}

pub fn save_image(image: Image, name: &str) { 
    let mut output_path = format!("outputs/{}.png", name);
    if std::path::Path::new(&output_path).exists() {
        let mut i = 1;
        while std::path::Path::new(&output_path).exists() {
            output_path = format!("outputs/{}_{}.png", name, i);
            i += 1;
        }
    }

    image::save_buffer(
        &output_path,
        &image.pixels,
        image.width,
        image.height,
        image::ColorType::Rgb8,
    ).expect("Failed to save image");
}


fn color_norm(a: &[u8], b: &[u8]) -> f32 {
    let norm = 8;
    ((0.2989 * (a[0] as f32 - b[0] as f32)).powi(norm) +
     (0.5870 * (a[1] as f32 - b[1] as f32)).powi(norm) +
     (0.1140 * (a[2] as f32 - b[2] as f32)).powi(norm)).powf(1.0 / norm as f32)
}


#[derive(Clone, Debug)]
struct Region{
    min_coord: Vec2D<u32>,
    max_coord: Vec2D<u32>,
}

impl Region {
    fn size(&self) -> u32 {
        (self.max_coord.x - self.min_coord.x) * (self.max_coord.y - self.min_coord.y)
    }

    fn min_dimension(&self) -> u32 {
        let width = self.max_coord.x - self.min_coord.x;
        let height = self.max_coord.y - self.min_coord.y;
        if width < height {
            width
        } else {
            height
        }
    }

    fn iter(&self) -> Vec<Vec2D<u32>> {
        let mut coords = Vec::new();
        for y in self.min_coord.y..self.max_coord.y {
            for x in self.min_coord.x..self.max_coord.x {
                coords.push(Vec2D::new(x, y));
            }
        }
        coords
    }

    fn lower_right_boundary_iter(&self) -> Vec<Vec2D<u32>> {
        let mut coords = Vec::new();
        for y in self.min_coord.y..self.max_coord.y - 1 {
            coords.push(Vec2D::new(self.max_coord.x - 1, y));
        }
        for x in self.min_coord.x..self.max_coord.x {
            coords.push(Vec2D::new(x, self.max_coord.y - 1));
        }
        coords
    }

}

struct Square {
    region: Region,
    color: Vec<u8>,
    loss: f32,
}

impl Square{
    fn new(image: &Image, region: Region) -> Self {
        let mut color = [0u32; 3];
        let mut count = 0;


        for coord in region.iter() {
            let pixel = image.get_pixel_from_coord(coord);
            color[0] += pixel[0] as u32;
            color[1] += pixel[1] as u32;
            color[2] += pixel[2] as u32;
            count += 1;
        }

        color[0] /= count;
        color[1] /= count;
        color[2] /= count;

        let color = [
            color[0] as u8,
            color[1] as u8,
            color[2] as u8,
        ];

        let mut loss = 0.0;

        for coord in region.iter() {
            let pixel = image.get_pixel_from_coord(coord);
            let diff = color_norm(&color, &pixel);
            loss += diff;
        }

        Self {
            region: region,
            color: color.to_vec(),
            loss: loss,
        }
        
    }

    fn divide(&self, image: &Image) -> [Square; 4] {
        let mid_x = (self.region.min_coord.x + self.region.max_coord.x) / 2;
        let mid_y = (self.region.min_coord.y + self.region.max_coord.y) / 2;

        let regions = [
            Region {
                min_coord: Vec2D::new(self.region.min_coord.x, self.region.min_coord.y),
                max_coord: Vec2D::new(mid_x, mid_y),
            },
            Region {
                min_coord: Vec2D::new(mid_x, self.region.min_coord.y),
                max_coord: Vec2D::new(self.region.max_coord.x, mid_y),
            },
            Region {
                min_coord: Vec2D::new(self.region.min_coord.x, mid_y),
                max_coord: Vec2D::new(mid_x, self.region.max_coord.y),
            },
            Region {
                min_coord: Vec2D::new(mid_x, mid_y),
                max_coord: Vec2D::new(self.region.max_coord.x, self.region.max_coord.y),
            },
        ];

        regions.map(|region| Square::new(image, region))

    }
}

enum Node {
    Leaf(Square),
    Internal {
        children: Box<[Node; 4]>,
    },
}

impl Node {
    fn draw_to_image(&self, image: &mut Image, show_outline: bool) {
        match self {
            Node::Leaf(square) => {
                for coord in square.region.iter() {
                    image.set_pixel_from_coord(coord, &square.color.clone());
                }
                if show_outline {
                    for coord in square.region.lower_right_boundary_iter() {
                        image.set_pixel_from_coord(coord, &[0, 0, 0]);
                    }
                }
            }
            Node::Internal { children } => {
                for child in children.iter() {
                    child.draw_to_image(image, show_outline);
                }
            }
        }
    }
}

pub struct QuadTree {
    root: Node,
    error_tolerance: f32,
    max_depth: usize,
}

impl QuadTree {

    pub fn new(image: &Image, error_tolerance: f32, max_depth: usize) -> Self {
        let region = Region {
            min_coord: Vec2D::new(0, 0),
            max_coord: Vec2D::new(image.width, image.height),
        };
        let root = Node::Leaf(Square::new(image, region));
        let mut quad_tree = QuadTree {
            root,
            error_tolerance,
            max_depth,
        };
        quad_tree.construct_top_down(image);
        quad_tree
    }

    fn construct_top_down_helper(
        node: &mut Node,
        error_tolerance: f32,
        image: &Image,
        region: Region,
        depth: usize,
        max_depth: usize,
    ) {
        match node {
            Node::Leaf(square) => {

                if depth < max_depth && square.loss / square.region.size() as f32 > error_tolerance && square.region.min_dimension() > 1 {
                    let children = square.divide(image).map(|square| {
                        Node::Leaf(square)
                    });
                    *node = Node::Internal { children: Box::new(children) };
                    Self::construct_top_down_helper(node, error_tolerance, image, region.clone(), depth, max_depth);
                }
                
            }
            Node::Internal { children } => {
                for child in children.iter_mut() {
                    Self::construct_top_down_helper(child, error_tolerance, image, region.clone(), depth + 1, max_depth);
                }
            }
        }
    }

    fn construct_top_down(&mut self, image: &Image) {
        let region = Region {
            min_coord: Vec2D::new(0, 0),
            max_coord: Vec2D::new(image.width, image.height),
        };
        let error_tolerance = self.error_tolerance;
        let max_depth = self.max_depth;
        Self::construct_top_down_helper(&mut self.root, error_tolerance, image, region, 0, max_depth);
    }
    
    pub fn approximate(&self, image: &Image, show_outline: bool) -> Vec<u8> {
        let mut image_copy = Image::new(image.pixels.clone(), image.width, image.height);
        self.root.draw_to_image(&mut image_copy, show_outline);
        
        let pixel_data = image_copy.pixels.to_vec(); 
        pixel_data
    }
}
