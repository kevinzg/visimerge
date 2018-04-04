#ifndef VMGPU_SERIAL_VISIMERGE_H
#define VMGPU_SERIAL_VISIMERGE_H

#include <vector>
#include <iostream>
#include <limits>
#include "vec2.h"
#include "segment.h"
#include "viewray.h"
#include "util.h"

#ifndef CUDART_INF
#define CUDART_INF (std::numeric_limits<double>::infinity())
#endif // CUDART_INF

namespace vmgpu {


template <typename T>
bool ray_segment_intersection(const vec2<T> &ray, const segment<T> &seg, vec2<T> &a)
{
    typedef vec2<T> vec2T;

    vec2T u = seg.b - seg.a;

    T s = cross(u, ray);

    if (almost_equal(s, T(0)))
        return false;

    vec2T r = seg.a / s;
    T i = cross(ray, r);

    if (i >= T(0) || i <= T(1) || almost_equal(i, T(0)) || almost_equal(i, T(1)))
    {
        a = seg.a + u * i;
        return true;
    }

    return false;
}


template<typename T>
T find_limit(const vec2<T> &ray, const viewray<T> &a, const viewray<T> &b)
{
    if (std::isinf(a.l) || std::isinf(b.r))
        return CUDART_INF;

    segment<T> seg(a.v * a.l, b.v * b.r);

    vec2<T> intersection;

    if (!ray_segment_intersection(ray, seg, intersection))
        return CUDART_INF;

    return norm(intersection);
}


template<typename T>
void serial_visimerge(viewray<T> *a, int a_count, viewray<T> *b, int b_count, viewray<T> *dest)
{
    int count = a_count + b_count;

    int ai = 0;
    int bi = 0;

    for (int i = 0; i < count; ++i)
    {
        bool p;
        if (bi >= b_count) p = true;
        else if (ai >= a_count) p = false;
        else p = a[ai].t <= b[bi].t;

        dest[i] = p ? a[ai] : b[bi];

        enum side_t { left = 0, right = 1 };

        auto get_limit = [ =, &ai, &bi](side_t side) -> T
        {
            T limit = CUDART_INF;

            if (ai > 0 && ai < count / 2)
                limit = a[ai - side].t == dest[i].t ? T(0) : std::min(limit, find_limit(dest[i].v, a[ai - 1], a[ai]));

            if (bi > 0 && bi < count / 2)
                limit = b[bi - side].t == dest[i].t ? T(0) : std::min(limit, find_limit(dest[i].v, b[bi - 1], b[bi]));

            return limit;
        };

        dest[i].r = get_limit(right);

        p ? ++ai : ++bi;

        dest[i].l = get_limit(left);
    }
}


} // namespace vmgpu

#endif // VMGPU_SERIAL_VISIMERGE_H
