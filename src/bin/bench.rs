//! In-process timer: runs each algorithm `warmup` + `runs` times at one size
//! and prints min / median / max wall time. Stand-in for `hyperfine` that
//! also avoids counting process start-up and PBM output.
//!
//! Usage: bench <width> <height> <runs> <algo>[,<algo>...]

use mandelbrot_lib::*;
use std::time::Instant;

fn algo(name: &str) -> Algorithm {
    match name {
        "scalar" => Algorithm::Scalar,
        #[cfg(feature = "nightly")]
        "simd" => Algorithm::Simd,
        "autovect" => Algorithm::AutoVectorization,
        "wide" => Algorithm::Wide,
        "wide8" => Algorithm::Wide8,
        "pulp" => Algorithm::Pulp,
        "ndarray" => Algorithm::Ndarray,
        other => panic!("unknown algorithm {:?}", other),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: usize = args[1].parse().unwrap();
    let h: usize = args[2].parse().unwrap();
    let runs: usize = args[3].parse().unwrap();
    let threads = rayon::current_num_threads();
    println!(
        "# {}x{} runs={} threads={} avx512f={} avx2={}",
        w,
        h,
        runs,
        threads,
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    let reference = Mandelbrot::generate((w, h), Algorithm::Scalar);
    for name in args[4].split(',') {
        let a = algo(name);
        let check = Mandelbrot::generate((w, h), a);
        assert!(check.data() == reference.data(), "{} disagrees with scalar", name);
        let mut t: Vec<f64> = (0..runs)
            .map(|_| {
                let s = Instant::now();
                let mb = Mandelbrot::generate((w, h), a);
                std::hint::black_box(&mb);
                s.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:<9} min {:>8.2} ms  median {:>8.2} ms  max {:>8.2} ms",
            name,
            t[0],
            t[t.len() / 2],
            t[t.len() - 1]
        );
    }
}
