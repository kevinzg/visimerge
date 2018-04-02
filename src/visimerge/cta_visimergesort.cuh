#ifndef VMGPU_CTA_VISIMERGESORT_CUH
#define VMGPU_CTA_VISIMERGESORT_CUH

#include <moderngpu/meta.hxx>
#include <moderngpu/cta_mergesort.hxx>
#include <moderngpu/cta_merge.hxx>
#include "cta_visimerge.cuh"
#include "viewray.h"

namespace vmgpu {


template<typename T>
MGPU_DEVICE viewray<T> *thread_visimergesort(viewray<T> *input, viewray<T> *buffer,
        const int vr_count, const int seg_count)
{
    int num_passes = mgpu::find_log2(seg_count);

    for (int p = 0; p < num_passes; ++p)
    {
        int sub_count = 2 << p;

        for (int begin = 0; begin < vr_count; begin += 2 * sub_count)
        {
            mgpu::merge_range_t range { 0, sub_count, sub_count, 2 * sub_count };
            serial_visimerge(input + begin, buffer + begin, range, sub_count, range);
        }

        mgpu::swap(input, buffer);
    }

    return input;
}


template<typename launch_t, typename T>
MGPU_DEVICE viewray<T> *merge_pass(viewray<T> *input, viewray<T> *dest, const mgpu::range_t vr_range,
                                   const mgpu::range_t seg_range, const int pass, const int tid)
{
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

    const int coop = 2 << pass;
    const mgpu::merge_range_t range = mgpu::compute_mergesort_range(2 * nv, tid, coop, 2 * vt);
    const int diag = 2 * vt * tid - range.a_begin;

    const int mp = mgpu::merge_path<mgpu::bounds_lower>(input, range, diag, [ = ](const viewray<T> &a, const viewray<T> &b) -> bool {
        return a.t < b.t;
    });

    serial_visimerge(input, dest + vr_range.begin, range.partition(mp, diag), vt, range);

    return dest;
}


template<typename launch_t, typename T>
MGPU_DEVICE viewray<T> *cta_visimergesort(viewray<T> *input, viewray<T> *buffer,
        const mgpu::range_t seg_range, const mgpu::range_t vr_range, const int tid)
{
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

    thread_visimergesort(input + vr_range.begin, buffer + vr_range.begin, vr_range.count(), seg_range.count());

    __syncthreads();

    const int num_passes = mgpu::find_log2(nt);

    for (int pass = 0; pass < num_passes; ++pass)
    {
        merge_pass<launch_t>(input, buffer, vr_range, seg_range, pass, tid);
        mgpu::swap(input, buffer);
        __syncthreads();
    }

    return input;
}


} // namespace vmgpu

#endif // VMGPU_CTA_VISIMERGESORT_CUH
