use std::{
    alloc::{alloc, dealloc, Layout},
    ops::{Deref, DerefMut},
    slice,
};

/// [0, n) の整数の集合を管理する定数倍が軽いデータ構造
///
/// https://topcoder-tomerun.hatenablog.jp/entry/2021/06/12/134643
#[derive(Debug, Clone)]
pub struct IndexSet {
    values: Vec<usize>,
    positions: Vec<Option<usize>>,
}

#[allow(dead_code)]
impl IndexSet {
    pub fn new(n: usize) -> Self {
        Self {
            values: vec![],
            positions: vec![None; n],
        }
    }

    pub fn add(&mut self, value: usize) {
        let pos = &mut self.positions[value];

        if pos.is_none() {
            *pos = Some(self.values.len());
            self.values.push(value);
        }
    }

    pub fn remove(&mut self, value: usize) {
        if let Some(index) = self.positions[value] {
            let last = *self.values.last().unwrap();
            self.values[index] = last;
            self.values.pop();
            self.positions[last] = Some(index);
            self.positions[value] = None;
        }
    }

    pub fn contains(&self, value: usize) -> bool {
        self.positions[value].is_some()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.values.iter()
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.values
    }
}

/// 32byte境界にアラインされた配列
/// 
/// 以下のサイトを参考にしたが、理解が怪しい。直してくれ～
/// https://qiita.com/moriai/items/67761b3c0d83da3b6bb5
#[derive(Debug)]
pub struct AlignedArrayU32 {
    ptr: *mut u32,
    layout: Layout,
    len: usize,
}

impl AlignedArrayU32 {
    pub fn new(len: usize) -> AlignedArrayU32 {
        let layout = match Layout::from_size_align(len * 4, 32) {
            Ok(layout) => layout,
            Err(err) => panic!("failed to create layout: {:?}", err),
        };

        let len = layout.size() / 4;

        unsafe {
            let ptr = alloc(layout);
            let ptr = ptr as *mut u32;
            AlignedArrayU32 { ptr, layout, len }
        }
    }

    pub fn as_slice(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedArrayU32 {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, self.layout) }
    }
}

impl Deref for AlignedArrayU32 {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for AlignedArrayU32 {
    fn deref_mut(&mut self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Clone for AlignedArrayU32 {
    fn clone(&self) -> Self {
        let mut result = AlignedArrayU32::new(self.layout.size() / 4);
        result.as_slice_mut().copy_from_slice(self.as_slice());
        result
    }
}
