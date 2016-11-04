-include config.mk

STATICLIB=libimagequant.a
SHAREDLIB=libimagequant.$(SOLIBSUFFIX)
SOVER=0

JNILIB=libimagequant.jnilib
DLL=libimagequant.dll
DLLIMP=libimagequant_dll.a
DLLDEF=libimagequant_dll.def

OBJS = pam.o mediancut.o blur.o mempool.o viter.o nearest.o libimagequant.o
SHAREDOBJS = $(subst .o,.lo,$(OBJS))

JAVACLASSES = org/pngquant/LiqObject.class org/pngquant/PngQuant.class org/pngquant/Image.class org/pngquant/Result.class
JAVAHEADERS = $(JAVACLASSES:.class=.h)
JAVAINCLUDE = -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I$(JAVA_HOME)/include/darwin

DISTFILES = $(OBJS:.o=.c) *.h README.md CHANGELOG COPYRIGHT Makefile configure
TARNAME = libimagequant-$(VERSION)
TARFILE = $(TARNAME)-src.tar.bz2

all: static

static: $(STATICLIB)

shared: $(SHAREDLIB)

dll:
	$(MAKE) CFLAGSADD="-DIMAGEQUANT_EXPORTS" $(DLL)

java: $(JNILIB)

$(DLL) $(DLLIMP): $(OBJS)
	$(CC) -fPIC -shared -o $(DLL) $^ $(LDFLAGS) -Wl,--out-implib,$(DLLIMP),--output-def,$(DLLDEF)

$(STATICLIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SHAREDOBJS):
	$(CC) -fPIC $(CFLAGS) -c $(@:.lo=.c) -o $@

$(SHAREDLIB): $(SHAREDOBJS)
	$(CC) -shared -Wl,-soname,$(SHAREDLIB).$(SOVER) -o $(SHAREDLIB).$(SOVER) $^ $(LDFLAGS)
	ln -fs $(SHAREDLIB).$(SOVER) $(SHAREDLIB)

$(OBJS): $(wildcard *.h) config.mk

$(JNILIB): $(JAVAHEADERS) $(STATICLIB) org/pngquant/PngQuant.c
	$(CC) -g $(CFLAGS) $(LDFLAGS) $(JAVAINCLUDE) -shared -o $@ $(STATICLIB) org/pngquant/PngQuant.c

$(JAVACLASSES): %.class: %.java
	javac $<

$(JAVAHEADERS): %.h: %.class
	javah -o $@ $(subst /,., $(patsubst %.class,%,$<)) && touch $@

dist: $(TARFILE)

$(TARFILE): $(DISTFILES)
	rm -rf $(TARFILE) $(TARNAME)
	mkdir $(TARNAME)
	cp $(DISTFILES) $(TARNAME)
	tar -cjf $(TARFILE) --numeric-owner --exclude='._*' $(TARNAME)
	rm -rf $(TARNAME)
	-shasum $(TARFILE)

clean:
	rm -f $(OBJS) $(SHAREDOBJS) $(SHAREDLIB).$(SOVER) $(SHAREDLIB) $(STATICLIB) $(TARFILE) $(DLL) $(DLLIMP) $(DLLDEF)
	rm -f $(JAVAHEADERS) $(JAVACLASSES) $(JNILIB)

distclean: clean
	rm -f config.mk

config.mk:
ifeq ($(filter %clean %distclean, $(MAKECMDGOALS)), )
	./configure
endif

.PHONY: all static shared clean dist distclean dll java
.DELETE_ON_ERROR:
