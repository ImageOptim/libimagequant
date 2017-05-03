extern crate gcc;
use std::env;
use std::path::PathBuf;

fn main() {
    if !std::process::Command::new("make").status().unwrap().success() {
        panic!("Download failed");
    }

    let mut out_dir: PathBuf = env::var_os("OUT_DIR").unwrap().into();
    out_dir.push("lib");

    gcc::compile_library("libimagequant.a", &[
        &out_dir.join("blur.c").to_str().unwrap(),
        &out_dir.join("kmeans.c").to_str().unwrap(),
        &out_dir.join("libimagequant.c").to_str().unwrap(),
        &out_dir.join("mediancut.c").to_str().unwrap(),
        &out_dir.join("mempool.c").to_str().unwrap(),
        &out_dir.join("nearest.c").to_str().unwrap(),
        &out_dir.join("pam.c").to_str().unwrap(),
    ]);
}
