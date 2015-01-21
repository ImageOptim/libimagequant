RUSTC ?= rustc

RUSTLIBSRC=src/lib.rs
LIQDIR=pngquant-2.2.0/lib
RUSTLIB=$(shell $(RUSTC) --crate-file-name $(RUSTLIBSRC))

all: crate example

crate: $(RUSTLIB)

$(RUSTLIB): $(RUSTLIBSRC) $(LIQDIR)/libimagequant.a
	$(RUSTC) -O -L $(LIQDIR) $<

$(LIQDIR)/libimagequant.a:: $(LIQDIR)
	make -C $(LIQDIR) static

$(LIQDIR):
	curl -L http://pngquant.org/pngquant-2.2.0-src.tar.bz2 | tar xj

example: $(RUSTLIB) examples/basic.rs
	$(RUSTC) -o $@ -L . examples/basic.rs
	@echo Run ./example

clean:
	rm -rf $(RUSTLIB) *.o example

.PHONY: all crate clean
