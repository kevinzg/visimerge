#ifndef VMGPU_SEGMENT_H
#define VMGPU_SEGMENT_H

#include <moderngpu/meta.hxx>
#include "vec2.h"

namespace vmgpu {


template<typename T>
struct segment
{
    vec2<T> a, b;

    MGPU_HOST_DEVICE segment(): a(), b() {}
    MGPU_HOST_DEVICE segment(vec2<T> a_, vec2<T> b_): a(a_), b(b_) {}

    MGPU_HOST_DEVICE vec2<T> operator[](size_t i) const { return i == 0 ? a : b; }
    MGPU_HOST_DEVICE vec2<T>& operator[](size_t i) { return i == 0 ? a : b; }
};


template<typename T>
MGPU_HOST_DEVICE bool almost_equal(const segment<T> &p, const segment<T> &q)
{
    return almost_equal(p.a, q.a) && almost_equal(p.b, q.b);
}


} // namespace vmgpu

#endif // VMGPU_SEGMENT_H
