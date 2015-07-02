LIQDIR=pngquant-2.5.0/lib

all: $(LIQDIR)/libimagequant.a

$(LIQDIR)/libimagequant.a:: $(LIQDIR)
	make -C $(LIQDIR) static

$(LIQDIR):
	curl -L http://pngquant.org/pngquant-2.5.0-src.tar.bz2 | tar xj

clean:
	make -C $(LIQDIR) clean

.PHONY: all clean
