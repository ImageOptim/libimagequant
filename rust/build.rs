//! This is a build script for Cargo https://crates.io/
//! It produces a static library that can be used by C or Rust.

extern crate gcc;

use std::env;
use std::path::PathBuf;

fn main() {
    let mut cc = gcc::Config::new();

    if env::var("PROFILE").map(|p|p != "debug").unwrap_or(true) {
        cc.define("NDEBUG", Some("1"));
    }

    if cfg!(target_arch="x86_64") ||
       (cfg!(target_arch="x86") && cfg!(feature = "sse")) {
        cc.define("USE_SSE", Some("1"));
    }

    if env::var("TARGET").unwrap().contains("windows-msvc") {
        cc.file("msvc-dist/libimagequant.c")
            .file("msvc-dist/nearest.c")
            .file("msvc-dist/kmeans.c")
            .file("msvc-dist/mediancut.c")
            .file("msvc-dist/mempool.c")
            .file("msvc-dist/pam.c")
            .file("msvc-dist/blur.c");
    } else {
        // This is so that I don't forget to publish MSVC version as well
        assert!(PathBuf::from("msvc-dist/libimagequant.c").exists());
        cc.flag("-std=c99");
        cc.file("libimagequant.c")
            .file("nearest.c")
            .file("kmeans.c")
            .file("mediancut.c")
            .file("mempool.c")
            .file("pam.c")
            .file("blur.c");
    }

    cc.compile("libimagequant.a");
}
