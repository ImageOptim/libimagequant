fn main() {
    println!("cargo:include={}", std::env::var("CARGO_MANIFEST_DIR").unwrap());
}
