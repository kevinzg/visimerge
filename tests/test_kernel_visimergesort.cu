#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include "visimerge/kernel_visimergesort.cuh"

using namespace vmgpu;


template<typename T>
std::vector<segment<T>> readfile(const std::string &filename)
{
    typedef segment<T> segmentT;

    std::ifstream file(filename.c_str());
    std::string line;

    std::vector<segmentT> vec;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);

        segmentT seg;

        ss >> seg.a.x;
        ss.get();
        ss >> seg.a.y;

        ss >> seg.b.x;
        ss.get();
        ss >> seg.b.y;

        vec.push_back(seg);
    }

    return vec;
}


template<typename T, typename Os>
void print_viewrays(const std::vector<viewray<T>> &vec, Os &out)
{
    for (int i = 0; i < vec.size(); ++i)
    {
        out << vec[i].t << " "
            << (almost_equal(vec[i].vx(), 0.0) ? 0.0 : vec[i].vx()) << " "
            << (almost_equal(vec[i].vy(), 0.0) ? 0.0 : vec[i].vy()) << " "
            << (std::isinf(vec[i].r) ? -1 : vec[i].r) << " "
            << (std::isinf(vec[i].l) ? -1 : vec[i].l) << std::endl;
    }
}


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << argv[0] << ": missing file operand" << std::endl;
        return EXIT_FAILURE;
    }

    auto vec = readfile<double>(argv[1]);

    if (1u << mgpu::find_log2(vec.size(), true) != vec.size())
    {
        std::cerr << "current visimergesort only works for segment sets with 2^k segments" << std::endl;
        return EXIT_FAILURE;
    }

    bool profile = argc >= 3 && std::string(argv[2]) == "--profile";

    std::vector<viewray<double>> vis(vec.size() * 2);

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

    if (!profile) print_viewrays(vis, std::cout);

    return 0;
}
