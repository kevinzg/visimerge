#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <cmath>
#include "visimerge/serial_visimergesort.h"
#include "visimerge/io_common.h"

using namespace vmgpu;


template<typename T>
int solve(const std::string &filename, bool profile = false)
{
    auto segments = readfile<T>(filename);

    if (1u << log2(segments.size(), true) != segments.size())
    {
        std::cerr << "current visimergesort only works for segment sets with 2^k segments" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<viewray<T>> vrays_vec(segments.size() * 2);

    serial_visimergesort(segments.data(), segments.size(), vrays_vec.data(), profile);

    if (!profile) print_viewrays(vrays_vec, std::cout);

    return EXIT_SUCCESS;
}


int main(int argc, char** argv)
{
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
        return solve<double>(argv[1], profile);

    return solve<float>(argv[1], profile);
}
