# [libimagequant](https://pngquant.org/lib/) — Image Quantization Library

Imagequant library converts RGBA images to palette-based 8-bit indexed images, *including* alpha component. It's ideal for generating tiny PNG images and [nice-looking GIFs](https://gif.ski).

Image encoding/decoding isn't handled by the library itself, bring your own encoder. If you're looking for a command-line tool, see [pngquant](https://pngquant.org).

## Getting started in C

This library can be used in C programs via [imagequant-sys](https://github.com/ImageOptim/libimagequant/tree/main/imagequant-sys) [Rust](https://www.rust-lang.org/) package.

```bash
rustup update
git clone https://github.com/ImageOptim/libimagequant
cd imagequant-sys
cargo build --release
# makes target/release/libimagequant_sys.a
```

See [the C library documentation for more details](https://pngquant.org/lib/).

## Getting started in Rust

Add to `Cargo.toml`:

```bash
rustup update
cargo add imagequant
```

[See docs.rs for the library API documentation](https://docs.rs/imagequant).

## License

Libimagequant is dual-licensed:

* For Free/Libre Open Source Software it's available under GPL v3 or later with additional [copyright notices](https://raw.github.com/ImageOptim/libimagequant/master/COPYRIGHT) for historical reasons.
* For use in closed-source software, AppStore distribution, and other non-GPL uses, you can [obtain a commercial license](https://supso.org/projects/pngquant). Feel free to ask kornel@pngquant.org for details and custom licensing terms if you need them.

## Upgrading instructions

libimagequant v2 used to be a C library. libimagequant v4 is written entirely in Rust, but still exports the same C interface for C programs. You will need to install Rust 1.70+ to build it, and adjust your build commands. If you do not want to upgrade, you can keep using [the C version of the library](https://github.com/imageoptim/libimagequant/tree/2.x) in the `2.x` branch of the [repo](https://github.com/ImageOptim/libimagequant).

### C static library users

Files for C/C++ are now in the `imagequant-sys/` subdirectory, not in the root of the repo. There is no `configure && make` any more.

To build the library, install [Rust via rustup](https://rustup.rs), and run:

```bash
rustup update
cd imagequant-sys
cargo build --release
```

It produces `target/release/libimagequant_sys.a` static library. The API, ABI, and header files remain the same, so everything else should work the same.
If you're building for macOS or iOS, see included xcodeproj file (add it as a [subproject](https://gitlab.com/kornelski/cargo-xcode#usage) to yours).

If you're building for Android, run `rustup target add aarch64-linux-android; cargo build --release --target aarch64-linux-android` and use `target/aarch64-linux-android/release/libimagequant_sys.a`. Same for cross-compiling to other platforms. See `rustup target list`.

### C dynamic library for package maintainers

If you're an application developer, please use the static linking option above — that option is much easier, and gives smaller executables.

The API and ABI of this library remains the same. It has the same sover, so it can be a drop-in replacement for the previous C version.

This library is now a typical Rust/Cargo library. If you want to set up [off-line builds](https://doc.rust-lang.org/cargo/faq.html#how-can-cargo-work-offline) or [override dependencies](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html), it works the same as for every other Rust project. See [Cargo docs](https://doc.rust-lang.org/cargo/) for things like [`cargo fetch`](https://doc.rust-lang.org/cargo/commands/cargo-fetch.html) or [`cargo vendor`](https://doc.rust-lang.org/cargo/commands/cargo-vendor.html) (but I don't recommend vendoring).

If you want to build a dynamic library, but aren't bothered by soname and rpath being wrong, modify `imagequant-sys/Cargo.toml` and add `"cdylib"` to the existing `crate-type` property, and then `cargo build --release` will do its usual half-finished job and build `target/release/libimagequant.{so,dylib,dll}`.

#### Building with `make`

`configure && make` is gone. I hoped I could build a dynamic library just by wrapping the static library, but apparently that won't work, so I can't easily recreate the old `make install`. I wish there was a more standard and lightweight solution than using the `cargo-c` tool, so if you're good at wrangling linker flags and symbol visibility, please send pull requests.

#### Building with `cargo-c`

The [`cargo-c`](https://lib.rs/cargo-c) tool knows how to build and link so/dylib properly, and generates an accurate pkg-config file, so it's de-facto required for a correct system-wide install of a dynamic library.

```bash
rustup update
cd imagequant-sys
cargo install cargo-c
cargo cinstall --prefix=/usr/local --destdir=.
```

This makes Rust 1.70 and `cargo-c` package a build-time dependency. No runtime deps (apart from Cargo-internal ones). OpenMP has been dropped entirely.

#### Interaction with pngquant

pngquant v2 can use this library as a dynamic library. However, pngquant v4 does not support unbundling. It uses this library as a Cargo dependency via its Rust-native interface. The shared libimagequant library exports only a stable ABI for C programs, and this interface is not useful for Rust programs.

### Upgrading for Rust users

If you've used the [`imagequant-sys`](https://lib.rs/imagequant-sys) crate, switch to the higher-level [`imagequant`](https://lib.rs/imagequant) crate. The `imagequant` v4 is almost entirely backwards-compatible, with small changes that the Rust compiler will point out (e.g. changed use of `c_int` to `u32`). See [docs](https://docs.rs/imagequant). Please fix any deprecation warnings you may get, because the deprecated functions will be removed.

The `openmp` Cargo feature has been renamed to `threads`.

`.new_image()` can now take ownership of its argument to avoid copying. If you get an error that `From<&Vec<RGBA>>` is not implemented, then either don't pass by reference (moves, avoids copying), or call `.as_slice()` on it (to copy the pixels), or use `.new_image_borrowed()` method instead.

### Threads support and WASM

By default, when the `threads` Cargo feature is enabled, this library uses multi-threading. Number of threads can be controlled via `RAYON_NUM_THREADS` environment variable.

Threads in WASM are experimental, and require [special handling](https://github.com/GoogleChromeLabs/wasm-bindgen-rayon). If you're targeting WASM, you'll most likely want to disable threads.

To disable threads when using this library as a dependency, disable default features like this in `Cargo.toml`:

```toml
[dependencies]
imagequant = { version = "4.0", default-features = false }
```

When you compile the library directly, add `--no-default-features` flag instead.

