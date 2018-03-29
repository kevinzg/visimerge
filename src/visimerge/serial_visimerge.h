#ifndef VMGPU_SERIAL_VISIMERGE_H
#define VMGPU_SERIAL_VISIMERGE_H

#include <vector>
#include <iostream>
#include "vec2.h"
#include "segment.h"
#include "viewray.h"
#include "util.h"

namespace vmgpu {


template <typename T>
bool ray_segment_intersection(const vec2<T> &ray, const segment<T> &seg, vec2<T> &a)
{
    typedef vec2<T> vec2T;

    vec2T u = seg.b - seg.a;

    T s = cross(u, ray);

    if (almost_equal(s, 0.0))
        return false;

    vec2T r = seg.a / s;
    T i = cross(ray, r);

    if ((almost_equal(i, 0.0) || i >= 0.0) || (almost_equal(i, 1.0) || i >= 1.0))
    {
        a = seg.a + u * i;
        return true;
    }

    return false;
}


template <typename T>
bool cut_segment(const segment<T> &seg, const vec2<T> &p, const vec2<T> &q, vec2<T> &a, vec2<T> &b)
{
    if (almost_equal(seg.a, seg.b))
        return false;

    if (!ray_segment_intersection(p, seg, a))
        return false;

    if (!ray_segment_intersection(q, seg, b))
        return false;

    return true;
}


template<typename T>
bool find_right_left_limit(const vec2<T> &p, const vec2<T> &q, const viewray<T> &a, const viewray<T> &b, T& l, T& r)
{
    if (a.l > 0 && b.r > 0)
    {
        segment<T> seg(a.v * a.l, b.v * b.r);

        vec2<T> c, d;

        if (!cut_segment(seg, p, q, c, d))
            return false;

        T nc = norm(c);
        T nd = norm(d);

        if (l + r < 0 || l + r > nc + nd)
        {
            l = nc;
            r = nd;

            return true;
        }
    }

    return false;
}


template<typename T>
void serial_visimerge(viewray<T> *a, size_t a_count, viewray<T> *b, size_t b_count, viewray<T> *dest)
{
    size_t count = a_count + b_count;

    size_t ai = 0;
    size_t bi = 0;

    for (size_t i = 0; i < count; ++i)
    {
        bool p;
        if (bi >= b_count) p = true;
        else if (ai >= a_count) p = false;
        else p = a[ai].t <= b[bi].t;

        dest[i] = p ? a[ai++] : b[bi++];

        if (i == 0) continue;

        size_t lai = p ? ai - 1 : ai;
        size_t lbi = p ? bi : bi - 1;

        T l = -1, r = -1;

        if (dest[i].t == dest[i - 1].t)
            l = r = 0;
        else
        {
            if (lai > 0 && lai < a_count)
                find_right_left_limit(dest[i - 1].v, dest[i].v, a[lai - 1], a[lai], l, r);

            if (lbi > 0 && lbi < b_count)
                find_right_left_limit(dest[i - 1].v, dest[i].v, b[lbi - 1], b[lbi], l, r);
        }

        dest[i - 1].l = l;
        dest[i].r = r;
    }
}


} // namespace vmgpu

#endif // VMGPU_SERIAL_VISIMERGE_H
