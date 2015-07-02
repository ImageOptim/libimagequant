#[libimagequant](http://pngquant.org/lib/) bindings for [Rust](http://www.rust-lang.org/)

Imagequant library converts RGBA images to 8-bit indexed images with palette, *including* alpha component. It's ideal for generating tiny PNG images (although image I/O isn't handled by the library itself).

This wrapper makes the library usable from Rust.

To build the `imagequant` crate:

    make crate

It will produce `libimagequant-….rlib`.
See [`example.rs`](https://github.com/pornel/libimagequant-rust/blob/master/example.rs) for usage.
For more details see [libimagequant documentation](http://pngquant.org/lib/).

