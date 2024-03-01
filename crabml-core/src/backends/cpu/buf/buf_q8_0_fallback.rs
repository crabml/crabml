use half::f16;

use super::BlockQ8_0;

pub fn quantize_f32_q8_0(data: &[f32]) -> Vec<BlockQ8_0> {
    let mut bs = Vec::with_capacity(data.len() / 32);

    for chunk in data.chunks(32) {
        let mut max_abs_value = 0.0;

        // Find the maximum absolute value in the chunk
        for &value in chunk {
            let abs_value = value.abs();
            if abs_value > max_abs_value {
                max_abs_value = abs_value;
            }
        }

        let d = max_abs_value / 127.0; // Compute the scaling factor
        let mut qs = [0_i8; 32]; // Initialize the quantized values array

        // Quantize the chunk
        for (i, &value) in chunk.iter().enumerate() {
            let scaled_value = value / d; // Scale the value
            // Convert the scaled value to i8, clamping it to the i8 range
            qs[i] = scaled_value.max(i8::MIN as f32).min(i8::MAX as f32) as i8;
        }

        // Store the block with the scaling factor and quantized values
        bs.push(BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        });
    }

    bs
}

pub fn vec_dot_q8_0_q8_0(abs: &[BlockQ8_0], bbs: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..bbs.len() {
        let mut sumi: i32 = 0;
        for j in 0..32 {
            sumi += (abs[i].qs[j] as i32) * (bbs[i].qs[j] as i32);
        }
        sumf += sumi as f32 * abs[i].d.to_f32() * bbs[i].d.to_f32();
    }

    sumf
}

