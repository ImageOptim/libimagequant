OUT_DIR?=vendor
LIQDIR=$(OUT_DIR)/lib

all: $(LIQDIR)

$(LIQDIR):
	mkdir -p "$(OUT_DIR)"
	curl -L https://pngquant.org/pngquant-2.9.1-src.tar.gz | tar xz -C "$(OUT_DIR)" --strip-components=1

clean:
	make -C $(LIQDIR) clean

.PHONY: all clean
.DELETE_ON_ERROR:
