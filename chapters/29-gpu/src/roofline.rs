//! What GPUs could do for inference, by the roofline model (chapter 4):
//! a kernel takes at least `bytes / bandwidth` and at least
//! `flops / peak`. These are estimates from published specifications,
//! not measurements.

/// A processor's limits.
#[derive(Clone, Copy, Debug)]
pub struct Device {
    pub name: &'static str,
    /// Dense 16-bit (bf16 or fp16) matrix throughput, in TFLOP/s.
    pub tflops: f64,
    /// Memory bandwidth, in GB/s.
    pub gb_per_s: f64,
    /// Memory, in GB.
    pub memory_gb: f64,
}

/// Published specifications (dense, no sparsity), and this course's CPU
/// as measured in chapters 4 and 23.
pub const DEVICES: [Device; 5] = [
    Device {
        name: "H100 SXM",
        tflops: 989.0,
        gb_per_s: 3350.0,
        memory_gb: 80.0,
    },
    Device {
        name: "A100 SXM 80 GB",
        tflops: 312.0,
        gb_per_s: 2039.0,
        memory_gb: 80.0,
    },
    Device {
        name: "RTX 4090",
        tflops: 165.0,
        gb_per_s: 1008.0,
        memory_gb: 24.0,
    },
    Device {
        name: "L4",
        tflops: 121.0,
        gb_per_s: 300.0,
        memory_gb: 24.0,
    },
    Device {
        name: "this CPU (4 cores)",
        tflops: 0.54,
        gb_per_s: 40.0,
        memory_gb: 15.0,
    },
];

/// A model's size, for decoding estimates.
#[derive(Clone, Copy, Debug)]
pub struct ModelSize {
    pub name: &'static str,
    pub params: f64,
}

impl Device {
    /// FLOPs per byte at which compute and memory take equal time.
    pub fn ridge(&self) -> f64 {
        self.tflops * 1e12 / (self.gb_per_s * 1e9)
    }

    /// The least time to decode one token at batch size 1 with `bf16`
    /// weights: reading every weight once.
    pub fn decode_floor(&self, m: &ModelSize) -> f64 {
        m.params * 2.0 / (self.gb_per_s * 1e9)
    }

    /// The batch size from which decoding is compute-bound: each `bf16`
    /// weight (2 bytes) does 2 FLOPs per sequence, 1 FLOP per byte.
    pub fn compute_bound_batch(&self) -> f64 {
        self.ridge()
    }

    /// The least time to prefill `tokens` tokens: 2 FLOPs per parameter
    /// per token, at peak.
    pub fn prefill_floor(&self, m: &ModelSize, tokens: f64) -> f64 {
        2.0 * m.params * tokens / (self.tflops * 1e12)
    }

    pub fn fits(&self, m: &ModelSize) -> bool {
        m.params * 2.0 / 1e9 <= self.memory_gb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_h100_numbers_come_out_as_expected() {
        let h100 = DEVICES[0];
        assert!((h100.ridge() - 295.2).abs() < 0.1);
        let llama8b = ModelSize {
            name: "8B",
            params: 8e9,
        };
        // 16 GB at 3.35 TB/s: 4.8 ms per token.
        assert!((h100.decode_floor(&llama8b) * 1e3 - 4.78).abs() < 0.01);
        assert!(h100.fits(&llama8b));
        assert!(!h100.fits(&ModelSize {
            name: "70B",
            params: 70e9
        }));
    }
}
