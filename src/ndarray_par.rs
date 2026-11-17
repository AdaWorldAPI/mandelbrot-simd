//! Vectorized parallel Mandelbrot implementation, using the `ndarray::simd`
//! facade of the AdaWorldAPI ndarray fork.
//!
//! The backend is chosen at compile time (`target-cpu`), not at run time:
//! on an AVX-512 build `F64x8` is one `__m512d`, on an AVX2 build it is a
//! pair of `__m256d` halves, elsewhere the scalar or NEON realization.

use crate::*;
use ndarray::simd::F64x8;
use std::mem::MaybeUninit;

#[allow(non_camel_case_types)]
type f64s = F64x8;
const LANES: usize = 8;

/// Returns the number of iterations it takes for the Mandelbrot sequence
/// to diverge at each lane, or `ITER_LIMIT` if it doesn't diverge.
#[inline(always)]
fn get_count(real: f64s, imag: f64s) -> [u64; LANES] {
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
    // The count is an exact integer in 0..=ITER_LIMIT held in an f64. Adding
    // 2^52 is exact for such values and leaves the integer in the low mantissa
    // bits, so the conversion is one vector add and a bit reinterpretation,
    // with no per-lane float-to-int cast.
    const TWO_52: f64 = 4_503_599_627_370_496.0;
    (count + f64s::splat(TWO_52)).to_bits().to_array()
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

    // Same allocation strategy as the wide and pulp arms: write into the
    // vector's uninitialized spare capacity, so no 41 MB zero-fill sits inside
    // the timed region.
    let mut out: Vec<u32> = Vec::with_capacity(width * height);
    out.spare_capacity_mut()[..width * height]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, row)| {
            let y = f64s::splat(yr.start + dy * (i as f64));
            // Fixed-size chunks (`as_chunks_mut`): each block is LANES slots,
            // written as one array expression.
            let (blocks, _) = row.as_chunks_mut::<LANES>();
            for (x, dst) in xs.iter().zip(blocks) {
                *dst = get_count(*x, y).map(|bits| MaybeUninit::new(bits as u32));
            }
        });
    // SAFETY: `width` is a multiple of LANES (asserted above), so the loop
    // above initialized every one of the first `width * height` elements, and
    // the capacity is at least that.
    unsafe { out.set_len(width * height) };
    out
}
