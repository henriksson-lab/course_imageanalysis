//! Reading a safetensors file (f16, bf16 or f32 tensors) into f32 vectors.
//! The header is a small JSON object; this module parses only what safetensors writes:
//! {"name": {"dtype": "F16", "shape": [..], "data_offsets": [a, b]}, "__metadata__": {..}}.
use std::collections::HashMap;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub type Weights = HashMap<String, Tensor>;

/// f16 bits -> f32 (handles subnormals, inf and NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalise
            let mut e: i32 = -1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = m & 0x3ff;
            sign | (((127 - 15 + 1 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ---- a minimal JSON reader (objects, arrays, strings, numbers, true/false/null) ----------

#[derive(Debug)]
enum Json {
    Obj(Vec<(String, Json)>),
    Arr(Vec<Json>),
    Str(String),
    Num(f64),
    Other,
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Result<u8, String> {
        self.ws();
        self.s.get(self.i).copied().ok_or_else(|| "unexpected end of header".to_string())
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek()? != c {
            return Err(format!("header: expected '{}' at {}", c as char, self.i));
        }
        self.i += 1;
        Ok(())
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        while self.i < self.s.len() {
            let c = self.s[self.i];
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    match e {
                        b'u' => {
                            let hex = std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|e| e.to_string())?;
                            let v = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            out.push(char::from_u32(v).unwrap_or('?'));
                            self.i += 4;
                        }
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        other => out.push(other as char),
                    }
                }
                _ => out.push(c as char),
            }
        }
        Err("unterminated string".into())
    }
    fn value(&mut self) -> Result<Json, String> {
        match self.peek()? {
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                if self.peek()? == b'}' {
                    self.i += 1;
                    return Ok(Json::Obj(v));
                }
                loop {
                    let k = self.string()?;
                    self.expect(b':')?;
                    let val = self.value()?;
                    v.push((k, val));
                    match self.peek()? {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Ok(Json::Obj(v));
                        }
                        _ => return Err(format!("header: bad object at {}", self.i)),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                if self.peek()? == b']' {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    v.push(self.value()?);
                    match self.peek()? {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Ok(Json::Arr(v));
                        }
                        _ => return Err(format!("header: bad array at {}", self.i)),
                    }
                }
            }
            b'"' => Ok(Json::Str(self.string()?)),
            _ => {
                let st = self.i;
                while self.i < self.s.len() && !b",}] \n\t\r".contains(&self.s[self.i]) {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[st..self.i]).map_err(|e| e.to_string())?;
                Ok(t.parse::<f64>().map(Json::Num).unwrap_or(Json::Other))
            }
        }
    }
}

fn get<'a>(o: &'a [(String, Json)], k: &str) -> Option<&'a Json> {
    o.iter().find(|(n, _)| n == k).map(|(_, v)| v)
}

fn nums(j: Option<&Json>) -> Result<Vec<usize>, String> {
    match j {
        Some(Json::Arr(a)) => a
            .iter()
            .map(|x| match x {
                Json::Num(n) => Ok(*n as usize),
                _ => Err("header: expected a number".to_string()),
            })
            .collect(),
        _ => Err("header: expected an array".into()),
    }
}

/// Parse a safetensors file. Float tensors (F16, BF16, F32) are converted to f32; other
/// dtypes (e.g. the I64 `num_batches_tracked` of batch norms) are skipped.
pub fn parse_safetensors(bytes: &[u8]) -> Result<Weights, String> {
    if bytes.len() < 8 {
        return Err("not a safetensors file (too short)".into());
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + n > bytes.len() {
        return Err("not a safetensors file (bad header length)".into());
    }
    let mut p = P { s: &bytes[8..8 + n], i: 0 };
    let root = match p.value()? {
        Json::Obj(o) => o,
        _ => return Err("header is not an object".into()),
    };
    let data = &bytes[8 + n..];
    let mut out = Weights::new();
    for (name, v) in root.iter() {
        if name == "__metadata__" {
            continue;
        }
        let o = match v {
            Json::Obj(o) => o,
            _ => continue,
        };
        let dtype = match get(o, "dtype") {
            Some(Json::Str(s)) => s.as_str(),
            _ => return Err(format!("{name}: no dtype")),
        };
        let shape = nums(get(o, "shape"))?;
        let off = nums(get(o, "data_offsets"))?;
        if off.len() != 2 || off[1] > data.len() || off[0] > off[1] {
            return Err(format!("{name}: bad data_offsets"));
        }
        let raw = &data[off[0]..off[1]];
        let vals: Vec<f32> = match dtype {
            "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            _ => continue,
        };
        let count: usize = shape.iter().product();
        if vals.len() != count {
            return Err(format!("{name}: {} values for shape {:?}", vals.len(), shape));
        }
        out.insert(name.clone(), Tensor { shape, data: vals });
    }
    Ok(out)
}
