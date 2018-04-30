#ifndef VMGPU_CTA_VISIMERGE_CUH
#define VMGPU_CTA_VISIMERGE_CUH

#include <moderngpu/meta.hxx>
#include <math_constants.h>
#include "segment.h"
#include "viewray.h"
#include "util.h"
#include "vec2.h"

namespace vmgpu {


template <typename T>
MGPU_DEVICE bool ray_segment_intersection(const vec2<T> &ray, const segment<T> &seg, vec2<T> &a)
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


template <typename T>
MGPU_DEVICE void serial_visimerge(const vray_array<T> &input, const vray_array<T> &dest,
                                  const mgpu::merge_range_t range, const int vt,
                                  const mgpu::merge_range_t merge_range)
{
    int ai = range.a_begin;
    int bi = range.b_begin;
    const int total_count = 2 * vt;

    for (int i = 0; i < total_count; ++i)
    {
        bool p;
        if (bi >= range.b_end) p = true;
        else if (ai >= range.a_end) p = false;
        else p = input.t[ai] <= input.t[bi];

        T dt = p ? input.t[ai] : input.t[bi];
        T dl = p ? input.l[ai] : input.l[bi];
        T dr = p ? input.r[ai] : input.r[bi];

        {
            T limit = CUDART_INF;

            const vec2<T> v(cos(dt), sin(dt));
            int j = -1;

            if (p && bi - merge_range.b_begin > 0 && bi - merge_range.b_begin < merge_range.b_count())
                j = bi;
            else if (!p && ai - merge_range.a_begin > 0 && ai - merge_range.a_begin < merge_range.a_count())
                j = ai;

            if (j >= 0 && !isinf(input.l[j - 1]) && !isinf(input.r[j]))
            {
                const vec2<T> av(cos(input.t[j - 1]), sin(input.t[j - 1]));
                const vec2<T> bv(cos(input.t[j]), sin(input.t[j]));

                segment<T> seg(av * input.l[j - 1], bv * input.r[j]);
                vec2<T> intersection;

                if (ray_segment_intersection(v, seg, intersection))
                    limit = norm(intersection);
            }

            dl = min(limit, dl);
            dr = min(limit, dr);
        }

        dest.t[i] = dt;
        dest.l[i] = dl;
        dest.r[i] = dr;

        p ? ++ai : ++bi;
    }
}


template <typename T>
MGPU_DEVICE void cta_visimerge(const vray_array<T> &input, const vray_array<T> &dest,
                               const mgpu::merge_range_t cta_range, const mgpu::merge_range_t merge_range,
                               const mgpu::range_t tile, const int tid, const int vt)
{
    const int diag = 2 * vt * tid;
    const int mp = mgpu::merge_path<mgpu::bounds_lower>(input.t, cta_range, diag, mgpu::less_t<T>());

    serial_visimerge(input, dest + tile.begin + diag, cta_range.partition(mp, diag), vt, merge_range);
};


} // namespace vmgpu

#endif // VMGPU_CTA_VISIMERGE_CUH
