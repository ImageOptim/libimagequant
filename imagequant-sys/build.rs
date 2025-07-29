fn main() {
    if cfg!(all(feature = "std", feature = "no_std")) {
        println!("cargo:warning=both std and no_std features are enabled in imagequant-sys");
    }
    println!("cargo:include={}", std::env::var("CARGO_MANIFEST_DIR").unwrap());
}
