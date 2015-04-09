fn main() {
    if !std::process::Command::new("make").status().unwrap().success() {
        panic!("Script failed");
    }
    println!("cargo:rustc-flags=-L pngquant-2.3.5/lib");
}
