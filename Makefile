OUT_DIR?=.
LIQDIR=$(OUT_DIR)/lib

all: $(LIQDIR)/libimagequant.a

$(LIQDIR)/libimagequant.a:: $(LIQDIR)
	make -C $(LIQDIR) static

$(LIQDIR):
	curl -L http://pngquant.org/pngquant-2.5.1-src.tar.bz2 | tar xj -C $(OUT_DIR) --strip-components=1

clean:
	make -C $(LIQDIR) clean

.PHONY: all clean
