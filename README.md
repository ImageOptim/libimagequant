# [libimagequant](http://pngquant.org/lib/) bindings for [Rust](http://www.rust-lang.org/)

Imagequant library converts RGBA images to 8-bit indexed images with palette, *including* alpha component.
It's ideal for generating tiny PNG images (although [image I/O](https://github.com/pornel/lodepng-rust) isn't handled by the library itself).

This wrapper makes the library usable from Rust.

Rust API closely follows the C API, but is slightly OO-ified:

    liq_set_dithering_level(result, 1.0);
      ↓
    result.set_dithering_level(1.0);

For more details see [libimagequant documentation](http://pngquant.org/lib/) and [Rust function reference](http://pornel.github.io/libimagequant-rust/imagequant/).

