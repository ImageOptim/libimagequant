extern crate gcc;
use std::env;
use std::path::PathBuf;

fn main() {
    if !std::process::Command::new("make").status().unwrap().success() {
        panic!("Download failed");
    }

    let mut liq_dir: PathBuf = env::var_os("OUT_DIR").unwrap().into();
    liq_dir.push("lib");

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
        .file(&liq_dir.join("blur.c").to_str().unwrap())
    	.file(&liq_dir.join("kmeans.c").to_str().unwrap())
        .file(&liq_dir.join("libimagequant.c").to_str().unwrap())
    	.file(&liq_dir.join("mediancut.c").to_str().unwrap())
        .file(&liq_dir.join("mempool.c").to_str().unwrap())
    	.file(&liq_dir.join("nearest.c").to_str().unwrap())
        .file(&liq_dir.join("pam.c").to_str().unwrap())
		.include(liq_dir)
        .compile("libimagequant.a");
}
