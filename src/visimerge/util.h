#ifndef VMGPU_UTIL_H
#define VMGPU_UTIL_H

#include <moderngpu/meta.hxx>
#include <algorithm>
#include <cmath>

namespace vmgpu {


template<typename T>
MGPU_HOST_DEVICE bool almost_equal(const T &a, const T &b)
{
    static const T EPS = 1e-9;

#ifdef __CUDACC__
    return fabs(a - b) <= EPS * fmax(1.0, fmax(fabs(a), fabs(b)));
#else
    return std::abs(a - b) <= EPS * std::max(1.0, std::max(std::abs(a), std::abs(b)));
#endif // __CUDACC__
}


} // namespace vmgpu

#endif // VMGPU_UTIL_H
