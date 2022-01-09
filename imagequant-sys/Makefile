# You can configure these
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
DESTDIR ?= ""

SOVER=0
VERSION=$(shell grep '^version = "4' Cargo.toml | grep -Eo "4\.[0-9.]+")
ifeq ($(shell uname -s),Darwin)
	SOLIBSUFFIX=dylib
	SHAREDLIBVER=libimagequant.$(SOVER).$(SOLIBSUFFIX)
	FIX_INSTALL_NAME=install_name_tool -id $(LIBDIR)/$(SHAREDLIBVER) $(DESTDIR)$(LIBDIR)/$(SHAREDLIBVER)
else
	SOLIBSUFFIX=so
	SHAREDLIBVER=libimagequant.$(SOLIBSUFFIX).$(SOVER)
	FIX_INSTALL_NAME=
endif
STATICLIB=libimagequant.a
SHAREDLIB=libimagequant.$(SOLIBSUFFIX)

JNILIB=libimagequant.jnilib

JAVACLASSES = org/pngquant/LiqObject.class org/pngquant/PngQuant.class org/pngquant/Image.class org/pngquant/Result.class
JAVAHEADERS = $(JAVACLASSES:.class=.h)
JAVAINCLUDE = -I'$(JAVA_HOME)/include' -I'$(JAVA_HOME)/include/linux' -I'$(JAVA_HOME)/include/win32' -I'$(JAVA_HOME)/include/darwin'

PKGCONFIG = imagequant.pc

all: static shared

static: $(STATICLIB)

shared: $(SHAREDLIB)

java: $(JNILIB)

target:: Cargo.toml
	cargo build --release

$(STATICLIB): target
	cp target/release/$(STATICLIB) $(STATICLIB)

libimagequant.so: $(STATICLIB)
	$(CC) -shared -Wl,-soname,$(SHAREDLIBVER) -o $(SHAREDLIBVER) $(STATICLIB) $(LDFLAGS)
	ln -fs $(SHAREDLIBVER) $(SHAREDLIB)

libimagequant.dylib: target
	$(CC) -shared -o $(SHAREDLIBVER) $(STATICLIB) $(LDFLAGS)
	ln -fs $(SHAREDLIBVER) $(SHAREDLIB)

$(JNILIB): $(JAVAHEADERS) $(STATICLIB) org/pngquant/PngQuant.c
	$(CC) -g $(CFLAGS) $(LDFLAGS) $(JAVAINCLUDE) -shared -o $@ org/pngquant/PngQuant.c $(STATICLIB)

$(JAVACLASSES): %.class: %.java
	javac $<

$(JAVAHEADERS): %.h: %.class
	javah -o $@ $(subst /,., $(patsubst %.class,%,$<)) && touch $@

example: example.c lodepng.h lodepng.c $(STATICLIB)
	$(CC) -g $(CFLAGS) -Wall example.c $(STATICLIB) -lm -o example

lodepng.h:
	curl -o lodepng.h -L https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h

lodepng.c:
	curl -o lodepng.c -L https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp

clean:
	rm -f $(SHAREDLIBVER) $(SHAREDLIB) $(STATICLIB)
	rm -f $(JAVAHEADERS) $(JAVACLASSES) $(JNILIB) example
	rm -rf target

distclean: clean
	rm -f imagequant.pc

install: all $(PKGCONFIG)
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(PKGCONFIGDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -m 644 $(STATICLIB) $(DESTDIR)$(LIBDIR)/$(STATICLIB)
	install -m 755 $(SHAREDLIBVER) $(DESTDIR)$(LIBDIR)/$(SHAREDLIBVER)
	ln -sf $(SHAREDLIBVER) $(DESTDIR)$(LIBDIR)/$(SHAREDLIB)
	install -m 644 $(PKGCONFIG) $(DESTDIR)$(PKGCONFIGDIR)/$(PKGCONFIG)
	install -m 644 libimagequant.h $(DESTDIR)$(INCLUDEDIR)/libimagequant.h
	$(FIX_INSTALL_NAME)

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(STATICLIB)
	rm -f $(DESTDIR)$(LIBDIR)/$(SHAREDLIBVER)
	rm -f $(DESTDIR)$(LIBDIR)/$(SHAREDLIB)
	rm -f $(DESTDIR)$(PKGCONFIGDIR)/$(PKGCONFIG)
	rm -f $(DESTDIR)$(INCLUDEDIR)/libimagequant.h

$(PKGCONFIG): Cargo.toml
	sed 's|@PREFIX@|$(PREFIX)|;s|@VERSION@|$(VERSION)|' < imagequant.pc.in > $(PKGCONFIG)

.PHONY: all static shared clean dist distclean dll java target
.DELETE_ON_ERROR:
