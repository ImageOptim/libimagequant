fn main() {
    cc::Build::new()
        .include("..")
        .file("test.c")
        .compile("imagequanttestbin");
}
