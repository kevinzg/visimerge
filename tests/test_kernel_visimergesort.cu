#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include "visimerge/kernel_visimergesort.cuh"
#include "visimerge/io_common.h"

using namespace vmgpu;


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << argv[0] << ": missing file operand" << std::endl;
        return EXIT_FAILURE;
    }

    auto vec = vmgpu::readfile<float>(argv[1]);

    if (1u << mgpu::find_log2(vec.size(), true) != vec.size())
    {
        std::cerr << "current visimergesort only works for segment sets with 2^k segments" << std::endl;
        return EXIT_FAILURE;
    }

    bool profile = argc >= 3 && std::string(argv[2]) == "--profile";

    std::vector<viewray<float>> vis(vec.size() * 2);

    mgpu::standard_context_t context(false);

    struct timeval start, end;

    if (profile) gettimeofday(&start, NULL);

    kernel_visimergesort(vec.data(), vec.size(), vis.data(), context, profile);

    if (profile)
    {
        gettimeofday(&end, NULL);

        long seconds  = end.tv_sec  - start.tv_sec;
        long useconds = end.tv_usec - start.tv_usec;

        double s = seconds + useconds / 1e6;

        std::cerr << "kernel_visimergesort took " << s * 1e3 << "ms to find the visibility region of " << vec.size()
                  << " segments" << std::endl;
    }

    if (!profile) vmgpu::print_viewrays(vis, std::cout);

    return 0;
}
