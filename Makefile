# Targetted to Tesla K80
ARCH=-gencode arch=compute_37,code=sm_37

OPTIONS=-std=c++11 -O2 -g -lineinfo --expt-extended-lambda -use_fast_math -Xptxas="-v" -I src/moderngpu/src

.PHONY: all tests clean

all: \
	tests

tests: \
	test_mergesort

test_mergesort: tests/test_mergesort.cu
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

clean:
	rm test_mergesort
