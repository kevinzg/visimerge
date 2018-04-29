#ifndef VMGPU_IO_COMMON_H
#define VMGPU_IO_COMMON_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "segment.h"
#include "viewray.h"
#include "vec2.h"

namespace vmgpu {


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
    auto flags = out.flags();
    out << std::fixed << std::setprecision(5);

    for (size_t i = 0; i < vec.size(); ++i)
    {
        out << vec[i].t << " "
            << vec[i].vx() << " "
            << vec[i].vy() << " "
            << (std::isinf(vec[i].r) ? -1 : vec[i].r) << " "
            << (std::isinf(vec[i].l) ? -1 : vec[i].l) << std::endl;
    }

    out.flags(flags);
}


} // namespace vmgpu

#endif // VMGPU_IO_COMMON_H
