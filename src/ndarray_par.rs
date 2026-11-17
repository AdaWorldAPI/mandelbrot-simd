//! Vectorized parallel Mandelbrot implementation, using the `ndarray::simd`
//! facade of the AdaWorldAPI ndarray fork.
//!
//! The backend is chosen at compile time (`target-cpu`), not at run time:
//! on an AVX-512 build `F64x8` is one `__m512d`, on an AVX2 build it is a
//! pair of `__m256d` halves, elsewhere the scalar or NEON realization.

use crate::*;
use ndarray::simd::F64x8;

type f64s = F64x8;
const LANES: usize = 8;

/// Returns the number of iterations it takes for the Mandelbrot sequence
/// to diverge at each lane, or `ITER_LIMIT` if it doesn't diverge.
#[inline(always)]
fn get_count(real: f64s, imag: f64s) -> [f64; LANES] {
    let mut cur_re = real;
    let mut cur_im = imag;
    let mut count = f64s::splat(0.0);
    let threshold = f64s::splat(THRESHOLD);
    let one = f64s::splat(1.0);
    let zero = f64s::splat(0.0);

    for _ in 0..ITER_LIMIT {
        let rr = cur_re * cur_re;
        let ii = cur_im * cur_im;

        let undiverged = (rr + ii).simd_le(threshold);
        if undiverged.to_bitmask() == 0 {
            break;
        }
        count += undiverged.select(one, zero);

        let ri = cur_re * cur_im;
        cur_re = real + (rr - ii);
        cur_im = imag + (ri + ri);
    }
    count.to_array()
}

pub fn generate(dims: Dimensions, xr: Range, yr: Range) -> Vec<u32> {
    let (width, height) = dims;
    assert_eq!(
        width % LANES,
        0,
        "image width = {} is not divisible by the number of vector lanes = {}",
        width,
        LANES,
    );
    let width_in_blocks = width / LANES;

    // The initial X values are the same for every row.
    let dx = (xr.end - xr.start) / (width as f64);
    let xs: Vec<f64s> = (0..width_in_blocks)
        .map(|b| {
            let mut lane = [0.0f64; LANES];
            for (k, x) in lane.iter_mut().enumerate() {
                *x = xr.start + dx * ((b * LANES + k) as f64);
            }
            f64s::from_array(lane)
        })
        .collect();

    let dy = (yr.end - yr.start) / (height as f64);
    let mut out = vec![0u32; width * height];

    out.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
        let y = f64s::splat(yr.start + dy * (i as f64));
        for (x, dst) in xs.iter().zip(row.chunks_exact_mut(LANES)) {
            let counts = get_count(*x, y);
            for (d, c) in dst.iter_mut().zip(counts.iter()) {
                *d = *c as u32;
            }
        }
    });
    out
}
