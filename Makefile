LIQDIR=pngquant-2.3.5/lib

all: $(LIQDIR)/libimagequant.a

$(LIQDIR)/libimagequant.a:: $(LIQDIR)
	make -C $(LIQDIR) static

$(LIQDIR):
	curl -L http://pngquant.org/pngquant-2.3.5-src.tar.bz2 | tar xj

clean:
	make -C $(LIQDIR) clean

.PHONY: all clean
