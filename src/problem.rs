use crate::grid::{Coord, Map2d};
use im_rc::HashMap;
use itertools::Itertools;
use proconio::{input, source::line::LineSource};
use rustc_hash::FxHashSet;
use std::{
    cmp::Reverse,
    io::{self, BufReader},
};

#[derive(Debug, Clone)]
pub struct Input {
    pub map_size: usize,
    pub oil_count: usize,
    pub eps: f64,
    pub oils: Vec<Oils>,
}

impl Input {
    fn new(map_size: usize, oil_count: usize, eps: f64, oils: Vec<Oils>) -> Self {
        let mut counts = HashMap::new();

        for oil in &oils {
            let entry = counts.entry(oil.pos.clone()).or_insert(0);
            *entry += 1;
        }

        Self {
            map_size,
            oil_count,
            eps,
            oils,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Oils {
    pub pos: Vec<Coord>,
    pub width: usize,
    pub height: usize,
    pub len: usize,
}

impl Oils {
    fn new(mut pos: Vec<Coord>) -> Self {
        pos.sort();
        let height = pos.iter().map(|c| c.row).max().unwrap() + 1;
        let width = pos.iter().map(|c| c.col).max().unwrap() + 1;
        let len = pos.len();

        Self {
            pos,
            width,
            height,
            len,
        }
    }
}

pub struct Judge {
    source: LineSource<BufReader<io::Stdin>>,
    wa: FxHashSet<Vec<Coord>>,
    comment_buf: Vec<String>,
    color_buf: Vec<Map2d<f64>>,
}

#[allow(dead_code)]
impl Judge {
    pub fn new() -> Self {
        let source = LineSource::new(BufReader::new(io::stdin()));
        Self {
            source,
            wa: FxHashSet::default(),
            comment_buf: vec![],
            color_buf: vec![],
        }
    }

    pub fn read_input(&mut self) -> Input {
        input! {
            from &mut self.source,
            n: usize,
            m: usize,
            eps: f64,
        }

        let mut oils = vec![];

        for _ in 0..m {
            input! {
                from &mut self.source,
                d: usize,
            }

            let mut pos = vec![];

            for _ in 0..d {
                input! {
                    from &mut self.source,
                    row: usize,
                    col: usize,
                }

                pos.push(Coord::new(row, col));
            }

            oils.push(Oils::new(pos));
        }

        oils.sort_by_key(|o| Reverse(o.len));

        Input::new(n, m, eps, oils)
    }

    pub fn query_single(&mut self, coord: Coord) -> i32 {
        self.flush_comments();

        println!("q 1 {} {}", coord.row, coord.col);

        input! {
            from &mut self.source,
            value: i32
        }

        value
    }

    pub fn query_multiple(&mut self, coords: &[Coord]) -> f64 {
        assert!(coords.len() >= 2);
        self.flush_comments();
        let len = coords.len();
        let coords = coords
            .iter()
            .map(|c| format!("{} {}", c.row, c.col))
            .join(" ");

        println!("q {} {}", len, coords);

        input! {
            from &mut self.source,
            value: f64
        }

        value
    }

    pub fn answer(&mut self, coords: &[Coord]) -> Result<(), ()> {
        self.flush_comments();
        let len = coords.len();
        let coords_vec = coords.iter().copied().collect_vec();

        if self.wa.contains(&coords_vec) {
            return Err(());
        }

        let coords_str = coords
            .iter()
            .map(|c| format!("{} {}", c.row, c.col))
            .join(" ");

        println!("a {} {}", len, coords_str);

        input! {
            from &mut self.source,
            value: i32
        }

        if value == 1 {
            Ok(())
        } else {
            self.wa.insert(coords_vec);
            Err(())
        }
    }

    pub fn comment(&mut self, message: &str) {
        self.comment_buf.push(message.to_string());
    }

    pub fn comment_colors(&mut self, colors: &Map2d<f64>) {
        self.color_buf.push(colors.clone());
    }

    fn flush_comments(&mut self) {
        for comment in &self.comment_buf {
            println!("# {}", comment);
        }

        self.comment_buf.clear();

        for colors in self.color_buf.iter() {
            for row in 0..colors.size {
                for col in 0..colors.size {
                    let c = Coord::new(row, col);
                    let v = ((1.0 - colors[c].clamp(0.0, 1.0)) * 255.0).round() as u8;
                    let color = format!("#{:02x}{:02x}{:02x}", 255, v, v);
                    println!("#c {row} {col} {color}");
                }
            }
        }

        self.color_buf.clear();
    }
}
