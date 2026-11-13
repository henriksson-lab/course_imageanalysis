//! Minimal in-memory reader for Keras weight files (HDF5): enough for `weights_best.h5` of the
//! pretrained StarDist models. The crate's own loader (`stardist_rs::weights::load_keras_hdf5_weights`)
//! opens a path with std::fs, which does not exist in the browser, so this reads the same
//! file from bytes and returns the same `KerasWeights` (kernels converted to OIHW with the
//! crate's `keras_conv2d_kernel_to_burn`).
//!
//! Supported: superblock version 0/1, version-1 object headers (with continuation blocks),
//! groups as symbol tables (B-tree version 1 + local heap), little-endian float32 datasets with
//! contiguous storage. Anything else (chunked or compressed data, other types) is an error.
use stardist_rs::weights::{keras_conv2d_kernel_to_burn, KerasWeight, KerasWeights};

pub struct H5<'a> {
    b: &'a [u8],
    so: usize, // size of offsets
    sl: usize, // size of lengths
}

type R<T> = Result<T, String>;

impl<'a> H5<'a> {
    fn u(&self, at: usize, n: usize) -> R<u64> {
        let s = self.b.get(at..at + n).ok_or_else(|| format!("h5: read past end at {at}"))?;
        let mut v = 0u64;
        for (i, &x) in s.iter().enumerate() {
            v |= (x as u64) << (8 * i);
        }
        Ok(v)
    }
    fn off(&self, at: usize) -> R<usize> {
        Ok(self.u(at, self.so)? as usize)
    }
    fn len(&self, at: usize) -> R<usize> {
        Ok(self.u(at, self.sl)? as usize)
    }
    fn sig(&self, at: usize, s: &[u8]) -> R<()> {
        if self.b.get(at..at + s.len()) == Some(s) {
            Ok(())
        } else {
            Err(format!("h5: expected {} at {at}", String::from_utf8_lossy(s)))
        }
    }

    pub fn new(b: &'a [u8]) -> R<Self> {
        if b.len() < 96 || &b[0..8] != b"\x89HDF\r\n\x1a\n" {
            return Err("h5: not an HDF5 file".into());
        }
        let ver = b[8];
        if ver > 1 {
            return Err(format!("h5: superblock version {ver} not supported"));
        }
        Ok(H5 { b, so: b[13] as usize, sl: b[14] as usize })
    }

    /// Address of the root group's object header.
    fn root(&self) -> R<usize> {
        // v0: 24 bytes of fixed fields, 4 addresses, then the root symbol table entry
        // (link name offset, object header address, ...). v1 has 4 more bytes before the addresses.
        let start = if self.b[8] == 0 { 24 } else { 28 };
        self.off(start + 4 * self.so + self.so)
    }

    /// Messages (type, data offset, size) of a version-1 object header.
    fn messages(&self, addr: usize) -> R<Vec<(u16, usize, usize)>> {
        if self.b.get(addr) != Some(&1) {
            return Err(format!("h5: object header at {addr} is not version 1"));
        }
        let n = self.u(addr + 2, 2)? as usize;
        let size = self.u(addr + 8, 4)? as usize;
        let mut blocks = vec![(addr + 16, size)];
        let mut out = Vec::new();
        let mut bi = 0;
        while bi < blocks.len() && out.len() < n {
            let (start, len) = blocks[bi];
            let mut p = start;
            while p + 8 <= start + len && out.len() < n {
                let t = self.u(p, 2)? as u16;
                let sz = self.u(p + 2, 2)? as usize;
                if t == 0x10 {
                    blocks.push((self.off(p + 8)?, self.len(p + 8 + self.so)?));
                }
                out.push((t, p + 8, sz));
                p += 8 + sz;
            }
            bi += 1;
        }
        Ok(out)
    }

    /// Children (name, object header address) of a group stored as a symbol table.
    fn children(&self, addr: usize) -> R<Vec<(String, usize)>> {
        let msgs = self.messages(addr)?;
        let (_, d, _) = msgs
            .iter()
            .find(|m| m.0 == 0x11)
            .ok_or_else(|| format!("h5: group at {addr} has no symbol table (new-style groups unsupported)"))?;
        let btree = self.off(*d)?;
        let heap = self.off(*d + self.so)?;
        self.sig(heap, b"HEAP")?;
        let heap_data = self.off(heap + 8 + 2 * self.sl)?;
        let mut out = Vec::new();
        self.walk_btree(btree, heap_data, &mut out)?;
        Ok(out)
    }

    fn walk_btree(&self, node: usize, heap_data: usize, out: &mut Vec<(String, usize)>) -> R<()> {
        self.sig(node, b"TREE")?;
        let level = self.b[node + 5];
        let used = self.u(node + 6, 2)? as usize;
        let mut p = node + 8 + 2 * self.so; // first key
        for _ in 0..used {
            p += self.sl; // key (heap offset)
            let child = self.off(p)?;
            p += self.so;
            if level > 0 {
                self.walk_btree(child, heap_data, out)?;
            } else {
                self.sig(child, b"SNOD")?;
                let n = self.u(child + 6, 2)? as usize;
                let entry = 2 * self.so + 24;
                for i in 0..n {
                    let e = child + 8 + i * entry;
                    let name_off = self.off(e)?;
                    let obj = self.off(e + self.so)?;
                    let s = heap_data + name_off;
                    let end = self.b[s..].iter().position(|&c| c == 0).map(|k| s + k).unwrap_or(s);
                    out.push((String::from_utf8_lossy(&self.b[s..end]).into_owned(), obj));
                }
            }
        }
        Ok(())
    }

    /// Shape and float32 values of a dataset, or None when the object is not a dataset.
    fn dataset(&self, addr: usize) -> R<Option<(Vec<usize>, Vec<f32>)>> {
        let msgs = self.messages(addr)?;
        let find = |t: u16| msgs.iter().find(|m| m.0 == t).map(|m| m.1);
        let (Some(space), Some(dtype), Some(layout)) = (find(0x1), find(0x3), find(0x8)) else {
            return Ok(None);
        };
        if find(0xB).is_some() {
            return Err("h5: filtered (compressed) datasets unsupported".into());
        }
        // Dataspace.
        let sver = self.b[space];
        let ndim = self.b[space + 1] as usize;
        let dims_at = if sver == 1 { space + 8 } else { space + 4 };
        let mut shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            shape.push(self.len(dims_at + i * self.sl)?);
        }
        // Datatype: class 1 (floating point), 4 bytes, little-endian.
        let class = self.b[dtype] & 0x0f;
        let size = self.u(dtype + 4, 4)? as usize;
        if class != 1 || size != 4 || self.b[dtype + 1] & 1 != 0 {
            return Err("h5: only little-endian float32 datasets supported".into());
        }
        // Layout: contiguous.
        let lver = self.b[layout];
        let data = if lver == 3 {
            if self.b[layout + 1] != 1 {
                return Err("h5: only contiguous storage supported".into());
            }
            self.off(layout + 2)?
        } else {
            if self.b[layout + 2] != 1 {
                return Err("h5: only contiguous storage supported".into());
            }
            self.off(layout + 8)?
        };
        let n: usize = shape.iter().product();
        let bytes = self
            .b
            .get(data..data + 4 * n)
            .ok_or_else(|| "h5: dataset past end of file".to_string())?;
        let values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Some((shape, values)))
    }
}

/// All float32 datasets of a Keras weights file, keyed by their path without the leading
/// slash (e.g. "conv2d_1/conv2d_1/kernel:0"), as `load_keras_hdf5_weights` does.
pub fn load_keras_weights(bytes: &[u8]) -> R<KerasWeights> {
    let f = H5::new(bytes)?;
    let mut weights = KerasWeights::default();
    let mut stack = vec![(String::new(), f.root()?)];
    while let Some((path, addr)) = stack.pop() {
        if let Some((shape, values)) = f.dataset(addr)? {
            let mut w = match shape.as_slice() {
                [kh, kw, ci, co] if path.ends_with("kernel:0") => {
                    keras_conv2d_kernel_to_burn([*kh, *kw, *ci, *co], &values)
                }
                _ => KerasWeight { name: String::new(), shape, values },
            };
            w.name = path.clone();
            weights.tensors.insert(path, w);
        } else {
            for (name, child) in f.children(addr)? {
                let p = if path.is_empty() { name } else { format!("{path}/{name}") };
                stack.push((p, child));
            }
        }
    }
    Ok(weights)
}
