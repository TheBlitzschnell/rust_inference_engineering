//! Chapter 9: reading and writing model weights in the safetensors format.
//!
//! A safetensors file is three things back to back:
//!
//! ```text
//! ┌──────────────┬──────────────────────────────┬──────────────────────────┐
//! │ N (8 bytes,  │ JSON header (N bytes)        │ raw tensor bytes         │
//! │ little-endian│ {"name": {"dtype": "BF16",   │ (every tensor's data,    │
//! │ u64)         │   "shape": [576, 576],       │  back to back, no gaps)  │
//! │              │   "data_offsets": [a, b]}, …}│                          │
//! └──────────────┴──────────────────────────────┴──────────────────────────┘
//! ```
//!
//! The parser in this crate never copies tensor data: [`SafeTensors`] and
//! [`TensorView`] borrow from the byte slice they were given, which is
//! usually a memory-mapped file ([`MappedFile`]). Because the file may come
//! from anywhere, every field of the header is validated before any tensor
//! is handed out.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

use ch02_numbers::{Bf16, F16};
use serde_json::Value;

/// The largest header we accept. The official implementation uses the same
/// limit: a real header is a few kilobytes to a few megabytes, and refusing
/// huge ones stops a malicious file from making us allocate gigabytes.
pub const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Element types that can appear in a safetensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E4M3,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl Dtype {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        }
    }

    /// The name used in the JSON header.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bool => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::F32 => "F32",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
            Dtype::F64 => "F64",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        const ALL: [Dtype; 14] = [
            Dtype::Bool,
            Dtype::U8,
            Dtype::I8,
            Dtype::F8E4M3,
            Dtype::I16,
            Dtype::U16,
            Dtype::F16,
            Dtype::BF16,
            Dtype::I32,
            Dtype::U32,
            Dtype::F32,
            Dtype::I64,
            Dtype::U64,
            Dtype::F64,
        ];
        ALL.into_iter().find(|d| d.name() == name)
    }
}

/// Everything that can be wrong with a safetensors file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The file is shorter than the 8-byte length prefix plus the header.
    Truncated { needed: u64, available: u64 },
    /// The header claims to be larger than [`MAX_HEADER_BYTES`].
    HeaderTooLarge(u64),
    /// The header is not valid UTF-8 JSON, or not a JSON object.
    InvalidHeader(String),
    /// One tensor's entry is malformed.
    InvalidTensor { name: String, reason: String },
    /// The data section is not exactly covered by the tensors: a gap, an
    /// overlap, or unused bytes at the end.
    InvalidLayout(String),
    /// A lookup for a tensor that is not in the file.
    NotFound(String),
    /// A view was requested in a type that does not match the tensor.
    WrongDtype {
        name: String,
        expected: Dtype,
        found: Dtype,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated { needed, available } => {
                write!(f, "file truncated: need {needed} bytes, have {available}")
            }
            Error::HeaderTooLarge(n) => write!(f, "header of {n} bytes exceeds the limit"),
            Error::InvalidHeader(why) => write!(f, "invalid header: {why}"),
            Error::InvalidTensor { name, reason } => write!(f, "tensor {name:?}: {reason}"),
            Error::InvalidLayout(why) => write!(f, "invalid data layout: {why}"),
            Error::NotFound(name) => write!(f, "no tensor named {name:?}"),
            Error::WrongDtype {
                name,
                expected,
                found,
            } => write!(f, "tensor {name:?} is {found:?}, not {expected:?}"),
        }
    }
}

impl std::error::Error for Error {}

/// Where one tensor lives, as described by the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range inside the data section (after the header).
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A parsed safetensors file that borrows its bytes from someone else.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    /// The data section: everything after the header.
    data: &'a [u8],
    /// Offset of `data` from the start of the file.
    data_offset: usize,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl<'a> SafeTensors<'a> {
    /// Parses and validates a whole file held in memory.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let available = bytes.len() as u64;
        let Some(prefix) = bytes.first_chunk::<8>() else {
            return Err(Error::Truncated {
                needed: 8,
                available,
            });
        };
        let header_len = u64::from_le_bytes(*prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(Error::HeaderTooLarge(header_len));
        }
        let data_offset = 8 + header_len;
        if data_offset > available {
            return Err(Error::Truncated {
                needed: data_offset,
                available,
            });
        }
        // Both casts are safe: data_offset <= bytes.len(), which is a usize.
        let header = &bytes[8..data_offset as usize];
        let data = &bytes[data_offset as usize..];

        let text = std::str::from_utf8(header)
            .map_err(|e| Error::InvalidHeader(format!("not UTF-8: {e}")))?;
        let json: Value = serde_json::from_str(text.trim_end_matches(' '))
            .map_err(|e| Error::InvalidHeader(format!("not JSON: {e}")))?;
        let Value::Object(entries) = json else {
            return Err(Error::InvalidHeader("not a JSON object".into()));
        };

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, entry) in entries {
            if name == "__metadata__" {
                metadata = parse_metadata(&entry)?;
            } else {
                let info = parse_tensor_info(&name, &entry)?;
                tensors.insert(name, info);
            }
        }
        check_layout(&tensors, data.len())?;
        Ok(Self {
            data,
            data_offset: data_offset as usize,
            tensors,
            metadata,
        })
    }

    /// Tensor names, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Free-form string metadata from the `__metadata__` entry.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Byte offset of the data section within the file. Tensor `t` starts
    /// at `data_offset() + t.start` bytes from the start of the file.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// A borrowed view of one tensor. The returned view lives as long as the
    /// underlying bytes (`'a`), not as long as `self`.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, Error> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_owned()))?;
        Ok(TensorView {
            name: name.to_owned(),
            dtype: info.dtype,
            shape: info.shape.clone(),
            data: &self.data[info.start..info.end],
        })
    }
}

fn parse_metadata(entry: &Value) -> Result<BTreeMap<String, String>, Error> {
    let Value::Object(map) = entry else {
        return Err(Error::InvalidHeader("__metadata__ is not an object".into()));
    };
    map.iter()
        .map(|(k, v)| match v {
            Value::String(s) => Ok((k.clone(), s.clone())),
            _ => Err(Error::InvalidHeader(format!(
                "__metadata__ value for {k:?} is not a string"
            ))),
        })
        .collect()
}

fn parse_tensor_info(name: &str, entry: &Value) -> Result<TensorInfo, Error> {
    let bad = |reason: &str| Error::InvalidTensor {
        name: name.to_owned(),
        reason: reason.to_owned(),
    };
    let dtype = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("missing \"dtype\""))?;
    let dtype = Dtype::from_name(dtype).ok_or_else(|| bad("unknown dtype"))?;

    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("missing \"shape\""))?
        .iter()
        .map(|d| {
            d.as_u64()
                .and_then(|d| usize::try_from(d).ok())
                .ok_or_else(|| bad("shape entries must be non-negative integers"))
        })
        .collect::<Result<Vec<usize>, _>>()?;

    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("missing \"data_offsets\""))?;
    let [start, end] = offsets.as_slice() else {
        return Err(bad("\"data_offsets\" must have two entries"));
    };
    let as_offset = |v: &Value| {
        v.as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| bad("offsets must be non-negative integers"))
    };
    let (start, end) = (as_offset(start)?, as_offset(end)?);
    if start > end {
        return Err(bad("data_offsets start after they end"));
    }

    // checked_mul: a hostile shape like [2^40, 2^40] must not overflow into a
    // small, plausible-looking size.
    let bytes = shape
        .iter()
        .try_fold(dtype.size(), |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad("shape is too large"))?;
    if end - start != bytes {
        return Err(bad(&format!(
            "shape {shape:?} of {dtype:?} needs {bytes} bytes, offsets give {}",
            end - start
        )));
    }
    Ok(TensorInfo {
        dtype,
        shape,
        start,
        end,
    })
}

/// The data section must be covered exactly: tensors sorted by offset must
/// start at 0, follow each other with no gap or overlap, and end at the end
/// of the file.
fn check_layout(tensors: &BTreeMap<String, TensorInfo>, data_len: usize) -> Result<(), Error> {
    let mut ranges: Vec<(usize, usize, &str)> = tensors
        .iter()
        .map(|(name, t)| (t.start, t.end, name.as_str()))
        .collect();
    ranges.sort_unstable();
    let mut expected = 0;
    for (start, end, name) in ranges {
        if start != expected {
            return Err(Error::InvalidLayout(format!(
                "tensor {name:?} starts at byte {start}, expected {expected} (gap or overlap)"
            )));
        }
        expected = end;
    }
    if expected != data_len {
        return Err(Error::InvalidLayout(format!(
            "tensors cover {expected} bytes but the data section has {data_len}"
        )));
    }
    Ok(())
}

/// One tensor's bytes, borrowed from the file, plus its type and shape.
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    fn expect_dtype(&self, expected: Dtype) -> Result<(), Error> {
        if self.dtype == expected {
            Ok(())
        } else {
            Err(Error::WrongDtype {
                name: self.name.clone(),
                expected,
                found: self.dtype,
            })
        }
    }

    /// The tensor as `&[Bf16]` without copying, if its bytes are 2-byte
    /// aligned in memory. Returns `Ok(None)` when they are not (the caller
    /// can then fall back to [`TensorView::to_f32_vec`] or copy).
    pub fn as_bf16(&self) -> Result<Option<&'a [Bf16]>, Error> {
        self.expect_dtype(Dtype::BF16)?;
        Ok(reinterpret::<Bf16>(self.data))
    }

    /// The tensor as `&[f32]` without copying, if 4-byte aligned.
    pub fn as_f32(&self) -> Result<Option<&'a [f32]>, Error> {
        self.expect_dtype(Dtype::F32)?;
        Ok(reinterpret::<f32>(self.data))
    }

    /// Converts the tensor to a new `Vec<f32>`, whatever its float type and
    /// alignment. Reads bytes one element at a time as little-endian, so it
    /// works on any machine.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, Error> {
        let pairs = || {
            self.data
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
        };
        match self.dtype {
            Dtype::F32 => Ok(self
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            Dtype::BF16 => Ok(pairs().map(|u| Bf16::from_bits(u).to_f32()).collect()),
            Dtype::F16 => Ok(pairs().map(|u| F16::from_bits(u).to_f32()).collect()),
            other => Err(Error::WrongDtype {
                name: self.name.clone(),
                expected: Dtype::F32,
                found: other,
            }),
        }
    }
}

/// Plain numeric types for which every bit pattern is a valid value, so a
/// byte slice may be reinterpreted as a slice of them.
///
/// # Safety
///
/// Implementors must have no padding and no invalid bit patterns.
pub unsafe trait Plain: Copy {}
// SAFETY: every 32-bit pattern is a valid f32 (possibly NaN); no padding.
unsafe impl Plain for f32 {}
// SAFETY: `Bf16` is `repr(transparent)` over `u16`; every pattern is valid.
unsafe impl Plain for Bf16 {}
// SAFETY: `F16` is `repr(transparent)` over `u16`; every pattern is valid.
unsafe impl Plain for F16 {}
// SAFETY: integers have no invalid bit patterns and no padding.
unsafe impl Plain for i8 {}
// SAFETY: as above.
unsafe impl Plain for u8 {}

/// Reinterprets little-endian bytes as `&[T]` when the address is aligned
/// for `T` and the length is a whole number of elements. Returns `None`
/// otherwise, or on a big-endian machine (where the bytes would need
/// swapping).
pub fn reinterpret<T: Plain>(bytes: &[u8]) -> Option<&[T]> {
    if cfg!(target_endian = "big") || !bytes.len().is_multiple_of(size_of::<T>()) {
        return None;
    }
    // SAFETY: `T: Plain` means every bit pattern is a valid `T`. `align_to`
    // only puts correctly aligned, whole elements in the middle slice, and we
    // accept the result only if nothing was left over at either end.
    let (head, body, tail) = unsafe { bytes.align_to::<T>() };
    (head.is_empty() && tail.is_empty()).then_some(body)
}

/// One tensor to be written by [`serialize`].
#[derive(Debug, Clone, Copy)]
pub struct TensorToWrite<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [usize],
    /// Raw little-endian bytes: `shape.product() * dtype.size()` of them.
    pub data: &'a [u8],
}

/// Builds a complete safetensors file in memory.
///
/// Tensors are written in the order given. The header is padded with spaces
/// to a multiple of 8 bytes, as the official writer does, so the data
/// section starts 8-byte aligned.
pub fn serialize(tensors: &[TensorToWrite<'_>], metadata: &BTreeMap<String, String>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let meta = metadata
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        header.insert("__metadata__".into(), Value::Object(meta));
    }
    let mut offset = 0;
    for t in tensors {
        let expected: usize = t.shape.iter().product::<usize>() * t.dtype.size();
        assert_eq!(
            t.data.len(),
            expected,
            "tensor {:?} has the wrong number of bytes",
            t.name
        );
        let entry = serde_json::json!({
            "dtype": t.dtype.name(),
            "shape": t.shape,
            "data_offsets": [offset, offset + t.data.len()],
        });
        header.insert(t.name.to_owned(), entry);
        offset += t.data.len();
    }
    let mut json = Value::Object(header).to_string().into_bytes();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + json.len() + offset);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    for t in tensors {
        out.extend_from_slice(t.data);
    }
    out
}

/// The little-endian bytes of an `f32` slice, for writing.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The little-endian bytes of a `bf16` slice, for writing.
pub fn bf16_bytes(values: &[Bf16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

/// Where the SmolLM2 files downloaded by `tools/download_model.sh` live:
/// `<models>/smollm2-<size>-instruct`, where `<models>` is `$INFER_MODELS`
/// if set and otherwise the `models/` directory at the root of this
/// repository. `size` is `"135m"` or `"360m"`.
pub fn smollm2_dir(size: &str) -> std::path::PathBuf {
    let models = std::env::var_os("INFER_MODELS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"),
        std::path::PathBuf::from,
    );
    models.join(format!("smollm2-{size}-instruct"))
}

/// A read-only memory-mapped file.
///
/// Mapping a file asks the operating system to make its contents appear in
/// our address space. Nothing is read up front: each 4 KB page is loaded
/// from disk (or from the OS page cache) the first time it is touched, and
/// the same physical pages are shared by every process mapping the file.
pub struct MappedFile {
    map: memmap2::Mmap,
}

impl MappedFile {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: memory-mapping is unsafe because the OS does not stop
        // another process from modifying or truncating the file while it is
        // mapped, which would change bytes behind a `&[u8]` (or make touching
        // them crash). We treat model files as read-only for the lifetime of
        // the process, which is the standard assumption of every inference
        // engine that maps its weights.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { map })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file() -> Vec<u8> {
        let a: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let b: Vec<Bf16> = (0..4).map(|i| Bf16::from_f32(i as f32 - 1.5)).collect();
        let c: Vec<u8> = vec![1, 2, 3];
        let (ab, bb) = (f32_bytes(&a), bf16_bytes(&b));
        let mut meta = BTreeMap::new();
        meta.insert("format".into(), "pt".into());
        serialize(
            &[
                TensorToWrite {
                    name: "a",
                    dtype: Dtype::F32,
                    shape: &[2, 3],
                    data: &ab,
                },
                TensorToWrite {
                    name: "b",
                    dtype: Dtype::BF16,
                    shape: &[4],
                    data: &bb,
                },
                TensorToWrite {
                    name: "c",
                    dtype: Dtype::U8,
                    shape: &[3],
                    data: &c,
                },
            ],
            &meta,
        )
    }

    /// Builds a file from a raw header string and data length.
    fn with_header(json: &str, data_len: usize) -> Vec<u8> {
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(json.as_bytes());
        out.resize(out.len() + data_len, 0);
        out
    }

    #[test]
    fn round_trip() {
        let file = sample_file();
        let st = SafeTensors::parse(&file).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!(st.metadata()["format"], "pt");
        assert_eq!(st.data_offset() % 8, 0, "header is padded to 8 bytes");

        let a = st.tensor("a").unwrap();
        assert_eq!(a.shape, [2, 3]);
        assert_eq!(a.to_f32_vec().unwrap(), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let b = st.tensor("b").unwrap().to_f32_vec().unwrap();
        assert_eq!(b, [-1.5, -0.5, 0.5, 1.5]);
        assert_eq!(st.tensor("c").unwrap().data, &[1, 2, 3]);
        assert!(matches!(st.tensor("zzz"), Err(Error::NotFound(_))));
        assert!(matches!(
            st.tensor("c").unwrap().to_f32_vec(),
            Err(Error::WrongDtype { .. })
        ));
    }

    #[test]
    fn zero_copy_views_agree_with_copies() {
        let file = sample_file();
        // Copy into a buffer we know is 8-byte aligned, as an mmap would be.
        let words: Vec<u64> = file
            .chunks(8)
            .map(|c| {
                let mut w = [0u8; 8];
                w[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(w)
            })
            .collect();
        let aligned: &[u8] = &reinterpret_u64_bytes(&words)[..file.len()];
        let st = SafeTensors::parse(aligned).unwrap();
        let a = st.tensor("a").unwrap();
        let view = a
            .as_f32()
            .unwrap()
            .expect("f32 data at offset 0 is aligned");
        assert_eq!(view, a.to_f32_vec().unwrap().as_slice());
        let b = st.tensor("b").unwrap();
        let view16 = b
            .as_bf16()
            .unwrap()
            .expect("bf16 data at an even offset is aligned");
        let from_view: Vec<f32> = view16.iter().map(|v| v.to_f32()).collect();
        assert_eq!(from_view, b.to_f32_vec().unwrap());
        // The view points into the file's bytes: no copy was made.
        assert!(std::ptr::eq(view.as_ptr().cast::<u8>(), a.data.as_ptr()));
    }

    fn reinterpret_u64_bytes(words: &[u64]) -> &[u8] {
        // SAFETY: u64 has no padding; viewing it as bytes is always valid.
        unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 8) }
    }

    #[test]
    fn misaligned_views_are_refused_not_faked() {
        let words = [0u64; 2];
        let bytes = reinterpret_u64_bytes(&words); // starts 8-byte aligned
        assert!(reinterpret::<f32>(&bytes[0..4]).is_some());
        assert!(
            reinterpret::<f32>(&bytes[1..5]).is_none(),
            "address is not a multiple of 4"
        );
        assert!(
            reinterpret::<f32>(&bytes[0..3]).is_none(),
            "3 bytes is not a whole f32"
        );
    }

    #[test]
    fn rejects_malformed_files() {
        use Error::*;
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (vec![1, 2, 3], "shorter than the length prefix"),
            ((u64::MAX).to_le_bytes().to_vec(), "absurd header size"),
            (with_header("{", 0), "invalid JSON"),
            (with_header("[1,2]", 0), "not an object"),
            (
                with_header(
                    r#"{"x":{"dtype":"F99","shape":[1],"data_offsets":[0,4]}}"#,
                    4,
                ),
                "unknown dtype",
            ),
            (
                with_header(
                    r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                    4,
                ),
                "shape/size mismatch",
            ),
            (
                with_header(
                    r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                    8,
                ),
                "gap at the start",
            ),
            (
                with_header(
                    r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                    8,
                ),
                "overlap",
            ),
            (
                with_header(
                    r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                    12,
                ),
                "unused bytes at the end",
            ),
            (
                with_header(
                    r#"{"x":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}"#,
                    4,
                ),
                "shape that overflows",
            ),
            (
                with_header(r#"{"__metadata__":{"k":1}}"#, 0),
                "non-string metadata",
            ),
        ];
        for (file, what) in cases {
            let err = SafeTensors::parse(&file).expect_err(what);
            let expected_kind = matches!(
                err,
                Truncated { .. }
                    | HeaderTooLarge(_)
                    | InvalidHeader(_)
                    | InvalidTensor { .. }
                    | InvalidLayout(_)
            );
            assert!(expected_kind, "{what}: unexpected error {err}");
        }
    }

    #[test]
    fn random_corruption_never_panics() {
        // Flip random bytes in a valid file many times. The parser must
        // return Ok or Err for every input; a panic here would be a crash in
        // a server that loads user-supplied files.
        let original = sample_file();
        let mut state = 0x1234_5678_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..5_000 {
            let mut file = original.clone();
            for _ in 0..=(next() % 4) {
                let i = (next() as usize) % file.len();
                file[i] = next() as u8;
            }
            if next() % 5 == 0 {
                file.truncate((next() as usize) % file.len());
            }
            if let Ok(st) = SafeTensors::parse(&file) {
                for name in st.names().map(str::to_owned).collect::<Vec<_>>() {
                    let t = st.tensor(&name).unwrap();
                    let _ = t.to_f32_vec();
                }
            }
        }
    }
}
