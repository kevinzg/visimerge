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

template<typename T>
std::vector<viewray<T>> solve_visibility_gpu(const std::vector<segment<T>> &host_segments, bool profile = false)
{
    mgpu::standard_context_t context(false);

    const int seg_count = host_segments.size();
    const int vr_count = 2 * seg_count;

    mgpu::mem_t<segment<T>> dev_segments(seg_count, context);
    vray_array<T> dev_vrays = vray_array<T>::create(vr_count, context);

    mgpu::htod(dev_segments.data(), host_segments.data(), seg_count);

    struct timeval start, end;

    if (profile) gettimeofday(&start, NULL);

    kernel_visimergesort(dev_segments, dev_vrays, context, profile);
    context.synchronize();

    if (profile)
    {
        gettimeofday(&end, NULL);

        long seconds  = end.tv_sec  - start.tv_sec;
        long useconds = end.tv_usec - start.tv_usec;

        double s = seconds + useconds / 1e6;

        std::cerr << "kernel_visimergesort took " << s * 1e3 << "ms to find the visibility region of " << seg_count
                  << " segments" << std::endl;
    }

    vray_array<T> host_vrays = vray_array<T>::create(vr_count, context, mgpu::memory_space_host);

    mgpu::dtoh(host_vrays.t, dev_vrays.t, vr_count);
    mgpu::dtoh(host_vrays.l, dev_vrays.l, vr_count);
    mgpu::dtoh(host_vrays.r, dev_vrays.r, vr_count);

    std::vector<viewray<T>> vrays_vec(vr_count);

    for (int i = 0; i < vr_count; ++i)
        vrays_vec[i] = { host_vrays.t[i], host_vrays.l[i], host_vrays.r[i] };

    vray_array<T>::destroy(dev_vrays, context);
    vray_array<T>::destroy(host_vrays, context, mgpu::memory_space_host);

    return vrays_vec;
}


int main(int argc, char** argv)
{
    typedef float real_t;

    if (argc < 2)
    {
        std::cerr << argv[0] << ": missing file operand" << std::endl;
        return EXIT_FAILURE;
    }

    auto segments = readfile<real_t>(argv[1]);

    if (1u << mgpu::find_log2(segments.size(), true) != segments.size())
    {
        std::cerr << "current visimergesort only works for segment sets with 2^k segments" << std::endl;
        return EXIT_FAILURE;
    }

    bool profile = argc >= 3 && std::string(argv[2]) == "--profile";

    auto vrays_vec = solve_visibility_gpu(segments, profile);

    if (!profile) print_viewrays(vrays_vec, std::cout);

    return 0;
}
