#ifndef VMGPU_UTIL_H
#define VMGPU_UTIL_H

#include <moderngpu/meta.hxx>
#include <algorithm>
#include <cmath>

using std::max;
using std::abs;

namespace vmgpu {


template<typename T>
MGPU_HOST_DEVICE bool almost_equal(const T &a, const T &b)
{
    static const T EPS = 1e-9;
    return abs(a - b) <= EPS * max(1.0, max(abs(a), abs(b)));
}


} // namespace vmgpu

#endif // VMGPU_UTIL_H
