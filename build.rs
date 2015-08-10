fn main() {
    if !std::process::Command::new("make").arg("-j8").status().unwrap().success() {
        panic!("Script failed");
    }
    println!("cargo:rustc-flags=-L {}/lib", std::env::var("OUT_DIR").unwrap());
}
