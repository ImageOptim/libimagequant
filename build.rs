extern crate gcc;
use std::env;

fn main() {
    assert!(std::path::Path::new("vendor/lib").exists(), "{}", env::current_dir().unwrap().display());

	let mut cfg = gcc::Config::new();
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        cfg.flag("-msse").define("USE_SSE", Some("1"));
    }
    if env::var("PROFILE").map(|x|x != "debug").unwrap_or(true)  {
        cfg.define("NDEBUG", Some("1"));
    }
    cfg
        .flag("-std=c99")
        .flag("-ffast-math")
        .file("vendor/lib/blur.c")
    	.file("vendor/lib/kmeans.c")
        .file("vendor/lib/libimagequant.c")
    	.file("vendor/lib/mediancut.c")
        .file("vendor/lib/mempool.c")
    	.file("vendor/lib/nearest.c")
        .file("vendor/lib/pam.c")
		.include("vendor/lib")
        .compile("libimagequant.a");
}
