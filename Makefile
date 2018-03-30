# Targetted to Tesla K80
ARCH=-gencode arch=compute_37,code=sm_37

OPTIONS=-std=c++11 -O2 -g -lineinfo --expt-extended-lambda -use_fast_math -Xptxas="-v" -I lib/moderngpu/src -I src

GCC_OPTIONS=-std=c++11 -O2 -Wall -Wno-unknown-pragmas -I lib/moderngpu/src -I src

.PHONY: all tests clean

all: \
	tests

tests: \
	test_mergesort \
	test_serial_visimergesort \
	test_kernel_visimergesort

test_mergesort: tests/test_mergesort.cu
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

test_serial_visimergesort: tests/test_serial_visimergesort.cpp src/visimerge/*.h
	g++ $(GCC_OPTIONS) -o $@ $<

test_kernel_visimergesort: tests/test_kernel_visimergesort.cu src/visimerge/*.h src/visimerge/*.cuh
	nvcc $(ARCH) $(OPTIONS) -o $@ $<

clean:
	rm test_mergesort
	rm test_serial_visimergesort
	rm test_kernel_visimergesort
