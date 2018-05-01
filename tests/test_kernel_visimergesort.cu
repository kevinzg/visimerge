#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <cmath>
#include "visimerge/kernel_visimergesort.cuh"
#include "visimerge/io_common.h"

using namespace vmgpu;


template<typename T, typename launch_t>
std::vector<viewray<T>> solve_visibility_gpu(const std::vector<segment<T>> &host_segments, bool profile = false)
{
    mgpu::standard_context_t context(false);

    const int seg_count = host_segments.size();
    const int vr_count = 2 * seg_count;

    mgpu::mem_t<segment<T>> dev_segments(seg_count, context);
    vray_array<T> dev_vrays = vray_array<T>::create(vr_count, context);

    mgpu::htod(dev_segments.data(), host_segments.data(), seg_count);

    kernel_visimergesort<T, launch_t>(dev_segments, dev_vrays, context, profile);
    context.synchronize();

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


template<typename T, typename launch_t>
int solve(const std::string &filename, bool profile = false)
{
    auto segments = readfile<T>(filename);

    if (1u << mgpu::find_log2(segments.size(), true) != segments.size())
    {
        std::cerr << "current visimergesort only works for segment sets with 2^k segments" << std::endl;
        return EXIT_FAILURE;
    }

    auto vrays_vec = solve_visibility_gpu<T, launch_t>(segments, profile);

    if (!profile) print_viewrays(vrays_vec, std::cout);

    return EXIT_SUCCESS;
}


int main(int argc, char** argv)
{
    typedef mgpu::launch_params_t<128, 1> float_launch;
    typedef mgpu::launch_params_t<64, 1> double_launch;

    if (argc < 2)
    {
        std::cerr << argv[0] << ": missing file operand" << std::endl;
        return EXIT_FAILURE;
    }

    std::set<std::string> options;
    for (int i = 2; i < argc; ++i)
        options.insert(std::string(argv[i]));

    bool profile = options.count("--profile");

    if (options.count("--double"))
        return solve<double, double_launch>(argv[1], profile);

    return solve<float, float_launch>(argv[1], profile);
}
