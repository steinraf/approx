use std::ops::Sub;

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

fn weighted_k_norm(v: &[f32], w: &[f32], norm: i32) -> f32 {
    // v dot w
    let vec = (0..3)
        .map(|i| v[i] * w[i])
        .collect::<Vec<f32>>();

        let max_abs = vec.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    
        if max_abs == 0.0 {
            return 0.0;
        }

        let sum = vec.iter()
            .map(|&x| (x / max_abs).powi(norm))
            .sum::<f32>();

        max_abs * sum.powf(1.0 / norm as f32)
}

fn color_norm(a: &[u8], b: &[u8]) -> f32 {
    let diff = (0..3).map(|i| (a[i] as f32 - b[i] as f32)).collect::<Vec<f32>>();

    diff.iter().map(|&x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(10000.0)
    // weighted_k_norm(&diff, &[0.2989, 0.5870, 0.1140], 8)
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

    fn extent(&self) -> Vec2D<u32> {
        Vec2D::new(
            self.max_coord.x - self.min_coord.x,
            self.max_coord.y - self.min_coord.y,
        )
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
    
    fn iter_with_stride(&self, stride: u32) -> Vec<Vec2D<u32>> {
        let mut coords = Vec::new();
        for y in (self.min_coord.y..self.max_coord.y).step_by(stride as usize) {
            for x in (self.min_coord.x..self.max_coord.x).step_by(stride as usize) {
                coords.push(Vec2D::new(x, y));
            }
        }
        coords
    }



    fn interior_iter(&self, subdivision: Option<u32>, border: Option<u32>) -> Vec<Vec2D<u32>> {
        let mut coords = Vec::new();
        let stride = match subdivision {
            Some(sub) => (self.min_dimension() / sub).max(1),
            None => 1,
        };
        let border = border.unwrap_or(1);
        for y in (self.min_coord.y + border..self.max_coord.y - (border - 1)).step_by(stride as usize) {
            for x in (self.min_coord.x + border..self.max_coord.x - (border - 1)).step_by(stride as usize) {
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

    fn center(&self) -> Vec2D<u32> {
        Vec2D::new(
            (self.min_coord.x + self.max_coord.x) / 2,
            (self.min_coord.y + self.max_coord.y) / 2,
        )
    }

    fn split_vertical(&self, x: u32) -> [Region; 2] {
        [
            Region {
                min_coord: self.min_coord.clone(),
                max_coord: Vec2D::new(x, self.max_coord.y),
            },
            Region {
                min_coord: Vec2D::new(x, self.min_coord.y),
                max_coord: self.max_coord.clone(),
            },
        ]
    }
    fn split_horizontal(&self, y: u32) -> [Region; 2] {
        [
            Region {
                min_coord: self.min_coord.clone(),
                max_coord: Vec2D::new(self.max_coord.x, y),
            },
            Region {
                min_coord: Vec2D::new(self.min_coord.x, y),
                max_coord: self.max_coord.clone(),
            },
        ]
    }
    fn split_quad(&self, x: u32, y: u32) -> [Region; 4] {
        [
            Region {
                min_coord: self.min_coord.clone(),
                max_coord: Vec2D::new(x, y),
            },
            Region {
                min_coord: Vec2D::new(x, self.min_coord.y),
                max_coord: Vec2D::new(self.max_coord.x, y),
            },
            Region {
                min_coord: Vec2D::new(self.min_coord.x, y),
                max_coord: Vec2D::new(x, self.max_coord.y),
            },
            Region {
                min_coord: Vec2D::new(x, y),
                max_coord: self.max_coord.clone(),
            },
        ]
    }

}


#[derive(Clone, Debug)]
pub enum DivisionPolicy {
    Regular,
    Adaptive(u32 /* subdivision */), // Adaptive division with a specified stride (min_square_dimension / subdivision)
}

struct Square {
    region: Region,
    color: Vec<u8>,
    loss: f32,
    division_policy: DivisionPolicy,
}

impl Square{
    fn new(image: &Image, region: Region, division_policy: DivisionPolicy) -> Self {
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
            region,
            color: color.to_vec(),
            loss,
            division_policy,
        }
        
    }


    fn divide_regular(&self, min_len: u32) -> Subdivision {
        if self.region.min_dimension() >= min_len * 2 {
            let (mid_x, mid_y) = self.region.center().into();
            return Subdivision::Quad(mid_x, mid_y);
        } else if self.region.extent().y < min_len * 2 && self.region.extent().x >= min_len * 2 {
            return Subdivision::Vertical(self.region.center().x);
        } else if self.region.extent().x < min_len * 2 && self.region.extent().y >= min_len * 2 {
            return Subdivision::Horizontal(self.region.center().y);
        } else {
            return Subdivision::None;
        }
    }

    fn best_horizontal_split(&self, image: &Image, stride: u32, min_len: u32) -> (u32, f32) {
        (self.region.min_coord.y + min_len..self.region.max_coord.y - (min_len - 1)).step_by(stride as usize)
            .map(|y| {

                let loss = self.region.split_horizontal(y).map(|region| {
                    Square::new(image, region, self.division_policy.clone()).loss
                }).iter().sum::<f32>();

                (y, loss)
            }).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())// panic on NaN
            .unwrap() 
    }

    fn best_vertical_split(&self, image: &Image, stride: u32, min_len: u32) -> (u32, f32) {
        (self.region.min_coord.x + min_len..self.region.max_coord.x - (min_len - 1)).step_by(stride as usize)
            .map(|x| {

                let loss = self.region.split_vertical(x).map(|region| {
                    Square::new(image, region, self.division_policy.clone()).loss
                }).iter().sum::<f32>();

                (x, loss)
            }).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())// panic on NaN
            .unwrap()
    }

    fn best_quad_split(&self, image: &Image, stride: u32, min_len: u32) -> (u32, u32, f32) {
        self.region.interior_iter(Some(stride), Some(min_len))
            .iter()
            .map(|mid_coord| {
                let (mid_x, mid_y) = (mid_coord.x, mid_coord.y);

                let loss  = self.region.split_quad(mid_x, mid_y).map(|region| Square::new(image, region, self.division_policy.clone()).loss).iter().sum::<f32>();
                (mid_x, mid_y, loss)

            }).min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) // panic on NaN
            .unwrap()
    }

    fn divide_adaptive(&self, image: &Image, subdivision: u32, min_len: u32) -> Subdivision {

        if self.region.min_dimension() >= min_len * 2 {
            let (hor_x, loss_x) = self.best_horizontal_split(image, subdivision, min_len);
            let (ver_y, loss_y) = self.best_vertical_split(image, subdivision, min_len);
            let (quad_x, quad_y, quad_loss) = self.best_quad_split(image, subdivision, min_len);
            let min_loss = loss_x.min(loss_y).min(quad_loss);
            if min_loss == loss_x {
                return Subdivision::Horizontal(hor_x);
            } else if min_loss == loss_y {
                return Subdivision::Vertical(ver_y);
            } else{
                return Subdivision::Quad(quad_x, quad_y);
            }
        } else if self.region.extent().x < min_len * 2 && self.region.extent().y >= min_len * 2 {
            let (y, _) = self.best_horizontal_split(image, subdivision, min_len);
            return Subdivision::Horizontal(y);
        } else if self.region.extent().x >= min_len * 2 && self.region.extent().y < min_len * 2 {
            let (x, _) = self.best_vertical_split(image, subdivision, min_len);
            return Subdivision::Vertical(x);
        } else {
            return Subdivision::None;
        }
    }



    fn divide(&self, image: &Image, min_len: u32) -> Option<InternalNode> {

        let subdivision = match self.division_policy {
            DivisionPolicy::Regular => self.divide_regular(min_len),
            DivisionPolicy::Adaptive(n) => self.divide_adaptive(image, n, min_len),
        };

        match subdivision {
            Subdivision::None => return None,
            Subdivision::Horizontal(y) => {
                let regions = self.region.split_horizontal(y);
                return Some(InternalNode::Double(Box::new([
                    Node::Leaf(Square::new(image, regions[0].clone(), self.division_policy.clone())),
                    Node::Leaf(Square::new(image, regions[1].clone(), self.division_policy.clone())),
                ])));
            }
            Subdivision::Vertical(x) => {
                let regions = self.region.split_vertical(x);
                return Some(InternalNode::Double(Box::new([
                    Node::Leaf(Square::new(image, regions[0].clone(), self.division_policy.clone())),
                    Node::Leaf(Square::new(image, regions[1].clone(), self.division_policy.clone())),
                ])));
            }
            Subdivision::Quad(x, y) => {
                let regions = self.region.split_quad(x, y);
                return Some(InternalNode::Quad(Box::new([
                    Node::Leaf(Square::new(image, regions[0].clone(), self.division_policy.clone())),
                    Node::Leaf(Square::new(image, regions[1].clone(), self.division_policy.clone())),
                    Node::Leaf(Square::new(image, regions[2].clone(), self.division_policy.clone())),
                    Node::Leaf(Square::new(image, regions[3].clone(), self.division_policy.clone())),
                ])));
            }
        }

    }
}


enum Subdivision {
    None,
    Horizontal(u32),
    Vertical(u32),
    Quad(u32, u32),
}

enum InternalNode {
    Double(Box<[Node; 2]>),
    Quad(Box<[Node; 4]>),
}

impl InternalNode {
    fn iter(&self) -> impl Iterator<Item = &Node> {
        match self {
            InternalNode::Double(children) => children.iter(),
            InternalNode::Quad(children) => children.iter(),
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        match self {
            InternalNode::Double(children) => children.iter_mut(),
            InternalNode::Quad(children) => children.iter_mut(),
        }
    }
}

enum Node {
    Leaf(Square),
    Internal {
        children: InternalNode,
    },
}

impl Node {
    fn draw_to_image(&self, image: &mut Image, show_outline: bool) {
        match self {
            Node::Leaf(square) => {
                for coord in square.region.iter() {
                    image.set_pixel_from_coord(coord, &square.color);
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
    leaf_count: usize,
    min_box_len: u32
}

impl QuadTree {

    pub fn new(image: &Image, error_tolerance: f32, max_depth: usize, division_policy: DivisionPolicy, min_box_len: u32) -> Self {
        let region = Region {
            min_coord: Vec2D::new(0, 0),
            max_coord: Vec2D::new(image.width, image.height),
        };
        let root = Node::Leaf(Square::new(image, region, division_policy));
        let mut quad_tree = QuadTree {
            root,
            error_tolerance,
            max_depth,
            leaf_count: 1,
            min_box_len,
        };
        quad_tree.construct_top_down(image);
        quad_tree
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    fn construct_top_down_helper(
        node: &mut Node,
        error_tolerance: f32,
        image: &Image,
        region: Region,
        depth_remaining: usize,
        node_counter: &mut usize,
        min_box_len: u32,
    ) {
        match node {
            Node::Leaf(square) => {

                if depth_remaining != 0 && square.loss / square.region.size() as f32 > error_tolerance {


                    let division = square.divide(image, min_box_len);
                    if division.is_none() {
                        return;
                    }
                  
                    
                    let children = division.unwrap();
                    *node = Node::Internal { children };
                    *node_counter += 3; // turned one leaf into 4
                    Self::construct_top_down_helper(node, error_tolerance, image, region.clone(), depth_remaining, node_counter , min_box_len);
                }
                
            }
            Node::Internal { children } => {
                for child in children.iter_mut() {
                    Self::construct_top_down_helper(child, error_tolerance, image, region.clone(), depth_remaining - 1, node_counter, min_box_len);
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
        Self::construct_top_down_helper(&mut self.root, error_tolerance, image, region, self.max_depth, &mut self.leaf_count, self.min_box_len);
    }
    
    pub fn approximate(&self, image: &Image, show_outline: bool) -> Vec<u8> {
        let mut image_copy = Image::new(vec![0; (image.width * image.height * 3) as usize], image.width, image.height);
        self.root.draw_to_image(&mut image_copy, show_outline);
        
        let pixel_data = image_copy.pixels.to_vec(); 
        pixel_data
    }
}
