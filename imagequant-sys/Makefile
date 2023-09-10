# You can configure these
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
DESTDIR ?= ""

VERSION=$(shell grep '^version = "4' Cargo.toml | grep -Eo "4\.[0-9.]+")
STATICLIB=libimagequant.a

JNILIB=libimagequant.jnilib

JAVACLASSES = org/pngquant/LiqObject.class org/pngquant/PngQuant.class org/pngquant/Image.class org/pngquant/Result.class
JAVAHEADERS = $(JAVACLASSES:.class=.h)
JAVAINCLUDE = -I'$(JAVA_HOME)/include' -I'$(JAVA_HOME)/include/linux' -I'$(JAVA_HOME)/include/win32' -I'$(JAVA_HOME)/include/darwin'

PKGCONFIG = imagequant.pc

all: static

static: $(STATICLIB)

java: $(JNILIB)

$(STATICLIB): Cargo.toml
	cargo build --release
	cp ../target/release/libimagequant_sys.a $(STATICLIB)

$(JNILIB): $(JAVAHEADERS) $(STATICLIB) org/pngquant/PngQuant.c
	# You may need to set LDFLAGS env var. See: cargo rustc -- --print native-static-libs
	$(CC) -g $(CFLAGS) $(LDFLAGS) $(JAVAINCLUDE) -shared -o $@ org/pngquant/PngQuant.c $(STATICLIB)

$(JAVACLASSES): %.class: %.java
	javac $<

$(JAVAHEADERS): %.h: %.class
	javah -o $@ $(subst /,., $(patsubst %.class,%,$<)) && touch $@

example: example.c lodepng.h lodepng.c $(STATICLIB)
	# remove -lpthread on Windows
	# add -ldl on Linux
	# You may need to set LDFLAGS env var. See: cargo rustc -- --print native-static-libs
	$(CC) -g $(CFLAGS) -Wall example.c $(STATICLIB) -lm -lpthread $(LDFLAGS) -o example

lodepng.h:
	curl -o lodepng.h -L https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h

lodepng.c:
	curl -o lodepng.c -L https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp

clean:
	rm -f $(SHAREDLIBVER) $(SHAREDLIB) $(STATICLIB)
	rm -f $(JAVAHEADERS) $(JAVACLASSES) $(JNILIB) example
	rm -rf ../target

distclean: clean
	rm -f imagequant.pc

install: all $(PKGCONFIG)
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(PKGCONFIGDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -m 644 $(STATICLIB) $(DESTDIR)$(LIBDIR)/$(STATICLIB)
	install -m 644 $(PKGCONFIG) $(DESTDIR)$(PKGCONFIGDIR)/$(PKGCONFIG)
	install -m 644 libimagequant.h $(DESTDIR)$(INCLUDEDIR)/libimagequant.h
	$(FIX_INSTALL_NAME)

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(STATICLIB)
	rm -f $(DESTDIR)$(PKGCONFIGDIR)/$(PKGCONFIG)
	rm -f $(DESTDIR)$(INCLUDEDIR)/libimagequant.h

$(PKGCONFIG): Cargo.toml
	sed 's|@PREFIX@|$(PREFIX)|;s|@VERSION@|$(VERSION)|' < imagequant.pc.in > $(PKGCONFIG)

.PHONY: all static clean distclean java
.DELETE_ON_ERROR:
