use std::{fmt::Display, vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Coord {
    pub row: usize,
    pub col: usize,
}

#[allow(dead_code)]
impl Coord {
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    pub fn in_map(&self, size: usize) -> bool {
        self.row < size && self.col < size
    }

    pub const fn to_index(&self, size: usize) -> CoordIndex {
        CoordIndex(self.row * size + self.col)
    }

    pub const fn dist(&self, other: &Self) -> usize {
        Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
    }

    const fn dist_1d(x0: usize, x1: usize) -> usize {
        (x0 as i64 - x1 as i64).abs() as usize
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct CoordDiff {
    pub dr: isize,
    pub dc: isize,
}

#[allow(dead_code)]
impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }

    pub const fn invert(&self) -> Self {
        Self::new(-self.dr, -self.dc)
    }
}

impl Display for CoordDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.dr, self.dc)
    }
}

impl std::ops::Add<CoordDiff> for Coord {
    type Output = Coord;

    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord::new(
            self.row.wrapping_add_signed(rhs.dr),
            self.col.wrapping_add_signed(rhs.dc),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct CoordIndex(pub usize);

#[allow(dead_code)]
impl CoordIndex {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn to_coord(&self, n: usize) -> Coord {
        Coord::new(self.0 / n, self.0 % n)
    }
}

#[allow(dead_code)]
pub const ADJACENTS: [CoordDiff; 4] = [
    CoordDiff::new(!0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(0, !0),
];

#[allow(dead_code)]
pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

#[derive(Debug, Clone)]
pub struct Map2d<T> {
    pub size: usize,
    map: Vec<T>,
}

#[allow(dead_code)]
impl<T> Map2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        assert_eq!(size * size, map.len());
        Self { size, map }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.map.iter_mut()
    }
}

#[allow(dead_code)]
impl<T: Clone> Map2d<T> {
    pub fn new_with(v: T, size: usize) -> Self {
        let map = vec![v; size * size];
        Self::new(map, size)
    }
}

impl<T> std::ops::Index<Coord> for Map2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self[coordinate.to_index(self.size)]
    }
}

impl<T> std::ops::IndexMut<Coord> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        let size = self.size;
        &mut self[coordinate.to_index(size)]
    }
}

impl<T> std::ops::Index<CoordIndex> for Map2d<T> {
    type Output = T;

    fn index(&self, index: CoordIndex) -> &Self::Output {
        &self.map[index.0]
    }
}

impl<T> std::ops::IndexMut<CoordIndex> for Map2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        &mut self.map[index.0]
    }
}
