//! Writes, parses and inspects safetensors files, and measures three ways
//! of loading SmolLM2's weights.
//!
//! Run with: cargo run --release -p ch09-safetensors
//! (the SmolLM2 part needs ./tools/download_model.sh first)

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use ch02_numbers::Bf16;
use ch09_safetensors::{
    Dtype, MappedFile, SafeTensors, TensorToWrite, bf16_bytes, f32_bytes, serialize, smollm2_dir,
};

fn main() {
    write_and_read_back();
    reject_bad_files();
    let path = smollm2_dir("135m").join("model.safetensors");
    if path.exists() {
        inspect(&path);
        load_times(&path);
    } else {
        println!("(skipping SmolLM2 parts: run ./tools/download_model.sh first)");
    }
}

/// Part 1: a tiny file, byte for byte.
fn write_and_read_back() {
    let weight: Vec<f32> = (0..6).map(|i| i as f32 / 10.0).collect();
    let bias: Vec<Bf16> = [1.0f32, -2.0].iter().map(|&v| Bf16::from_f32(v)).collect();
    let (wb, bb) = (f32_bytes(&weight), bf16_bytes(&bias));
    let mut meta = BTreeMap::new();
    meta.insert("note".to_string(), "chapter 9 demo".to_string());
    let file = serialize(
        &[
            TensorToWrite {
                name: "layer.weight",
                dtype: Dtype::F32,
                shape: &[2, 3],
                data: &wb,
            },
            TensorToWrite {
                name: "layer.bias",
                dtype: Dtype::BF16,
                shape: &[2],
                data: &bb,
            },
        ],
        &meta,
    );
    let header_len = u64::from_le_bytes(file[..8].try_into().expect("8 bytes"));
    println!("== 1. a two-tensor file: {} bytes in total", file.len());
    println!(
        "   first 8 bytes (header length, little-endian): {:?} = {header_len}",
        &file[..8]
    );
    println!(
        "   header: {}",
        String::from_utf8_lossy(&file[8..8 + header_len as usize]).trim_end()
    );
    println!(
        "   data section: {} bytes",
        file.len() - 8 - header_len as usize
    );
    let st = SafeTensors::parse(&file).expect("valid file");
    for name in st.names() {
        let t = st.tensor(name).expect("listed tensor");
        println!(
            "   {name:<13} {:?} {:?} -> {:?}",
            t.dtype,
            t.shape,
            t.to_f32_vec().expect("float tensor")
        );
    }
    println!();
}

/// Part 2: what the validator says about broken files.
fn reject_bad_files() {
    println!("== 2. rejecting broken files");
    let header = |json: &str, data: usize| {
        let mut f = (json.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(json.as_bytes());
        f.resize(f.len() + data, 0);
        f
    };
    let cases = [
        ("truncated file", vec![7u8, 0, 0]),
        ("header size of 2^40", (1u64 << 40).to_le_bytes().to_vec()),
        (
            "shape does not match byte range",
            header(
                r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,4]}}"#,
                4,
            ),
        ),
        (
            "two tensors overlap",
            header(
                r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#,
                4,
            ),
        ),
    ];
    for (label, bytes) in cases {
        match SafeTensors::parse(&bytes) {
            Ok(_) => println!("   {label:<32} accepted (unexpected!)"),
            Err(e) => println!("   {label:<32} -> {e}"),
        }
    }
    println!();
}

/// Part 3: what is inside SmolLM2-135M's weight file.
fn inspect(path: &std::path::Path) {
    let file = MappedFile::open(path).expect("open model file");
    let st = SafeTensors::parse(file.bytes()).expect("valid safetensors");
    let mut params = 0usize;
    let mut by_dtype: BTreeMap<&str, usize> = BTreeMap::new();
    let mut aligned = [0usize; 3]; // 2, 4, 64 bytes
    for name in st.names() {
        let info = st.info(name).expect("listed");
        params += info.num_elements();
        *by_dtype.entry(info.dtype.name()).or_default() += 1;
        let start = st.data_offset() + info.start;
        for (slot, align) in aligned.iter_mut().zip([2, 4, 64]) {
            if start % align == 0 {
                *slot += 1;
            }
        }
    }
    println!("== 3. {}", path.display());
    println!(
        "   {} tensors, {params} parameters ({:.1} M), dtypes {by_dtype:?}",
        st.len(),
        params as f64 / 1e6
    );
    println!(
        "   header: {} bytes; data starts at byte {}",
        st.data_offset() - 8,
        st.data_offset()
    );
    println!("   first tensors:");
    for name in st.names().take(6) {
        let i = st.info(name).expect("listed");
        println!("     {name:<48} {:?} {:?}", i.dtype, i.shape);
    }
    println!(
        "   tensors whose data starts 2-byte aligned: {}, 4-byte: {}, 64-byte: {} (of {})",
        aligned[0],
        aligned[1],
        aligned[2],
        st.len()
    );
    println!();
}

/// Part 4: three ways to get the weights into memory.
fn load_times(path: &std::path::Path) {
    println!("== 4. loading the weights (file already in the OS page cache)");

    let start = Instant::now();
    let bytes = std::fs::read(path).expect("read file");
    let read = start.elapsed();
    println!(
        "   std::fs::read of {:.0} MB:        {read:>9.2?}  ({:.1} GB/s)",
        bytes.len() as f64 / 1e6,
        bytes.len() as f64 / read.as_secs_f64() / 1e9
    );
    drop(bytes);

    let start = Instant::now();
    let file = MappedFile::open(path).expect("map file");
    let st = SafeTensors::parse(file.bytes()).expect("valid");
    let mapped = start.elapsed();
    println!("   mmap + parse header:              {mapped:>9.2?}");

    let start = Instant::now();
    let mut checksum = 0u64;
    for name in st.names() {
        let t = st.tensor(name).expect("listed");
        // Touch one byte per 4 KB page: enough to fault every page in.
        checksum += t
            .data
            .iter()
            .step_by(4096)
            .map(|&b| u64::from(b))
            .sum::<u64>();
    }
    black_box(checksum);
    println!(
        "   first touch of every page:        {:>9.2?}",
        start.elapsed()
    );

    let start = Instant::now();
    let mut zero_copy = 0;
    for name in st.names() {
        if matches!(st.tensor(name).expect("listed").as_bf16(), Ok(Some(_))) {
            zero_copy += 1;
        }
    }
    println!(
        "   zero-copy &[Bf16] views:          {:>9.2?}  ({zero_copy} of {} tensors)",
        start.elapsed(),
        st.len()
    );

    let start = Instant::now();
    let mut total = 0usize;
    let converted: Vec<Vec<f32>> = st
        .names()
        .map(|name| {
            let v = st
                .tensor(name)
                .expect("listed")
                .to_f32_vec()
                .expect("float");
            total += v.len() * 4;
            v
        })
        .collect();
    black_box(&converted);
    println!(
        "   convert everything to Vec<f32>:   {:>9.2?}  ({:.0} MB allocated)",
        start.elapsed(),
        total as f64 / 1e6
    );
}
