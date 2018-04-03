#ifndef VMGPU_KERNEL_VISIMERGESORT_H
#define VMGPU_KERNEL_VISIMERGESORT_H

#include <iostream>
#include <algorithm>
#include <math_constants.h>
#include <moderngpu/transform.hxx>
#include <moderngpu/intrinsics.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/types.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include "cta_visimerge.cuh"
#include "cta_visimergesort.cuh"
#include "vec2.h"

namespace vmgpu {


template<typename T>
void init_viewrays(segment<T> *segs, int count, viewray<T> *dest, mgpu::context_t &context, bool profile = false)
{
    typedef viewray<T> viewrayT;
    typedef vec2<T> vec2T;

    auto k = [] MGPU_DEVICE (int i, segment<T> *segs, viewray<T> *dest)
    {
        vec2T &a = segs[i].a;
        vec2T &b = segs[i].b;

        if (cross(a, b) < 0)
            mgpu::swap(a, b);

        T atan2b = atan2(b);
        atan2b = atan2b == 0 ? 2 * M_PI : atan2b;

        dest[2 * i] = viewrayT(atan2(a), norm(a), CUDART_INF);
        dest[2 * i + 1] = viewrayT(atan2b, CUDART_INF, norm(b));
    };

    if (profile) context.timer_begin();

    mgpu::transform(k, count, context, segs, dest);
    context.synchronize();

    if (profile)
    {
        double s = context.timer_end();
        std::cerr << "init_viewrays took " << s * 1e3 << "ms to convert " << count << " segments ("
                  << count * sizeof(segment<T>) / double(1 << 30) / s << " GiB/s)" << std::endl;
    }
}


template<typename T>
void kernel_visimergesort(segment<T> *segs, const int seg_count, viewray<T> *output, mgpu::context_t &context,
                          bool profile = false)
{
    typedef segment<T> segmentT;
    typedef viewray<T> viewrayT;

    const int vr_count = seg_count * 2;

    mgpu::mem_t<segmentT> dev_segments(seg_count, context);
    mgpu::htod(dev_segments.data(), segs, seg_count);

    mgpu::mem_t<viewrayT> dev_viewrays(vr_count, context);
    init_viewrays(dev_segments.data(), seg_count, dev_viewrays.data(), context, profile);

    mgpu::mem_t<viewrayT> dev_buffer(vr_count, context);

    typedef mgpu::launch_params_t<16, 16> launch_t;

    const int nv = launch_t::nv(context);
    const int num_ctas = mgpu::div_up(seg_count, nv);
    const int num_passes = mgpu::find_log2(num_ctas, true);

    auto k = [] MGPU_DEVICE (const int tid, const int cta, viewray<T> *input, const int seg_count, viewray<T> *buffer)
    {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        // Global memory
        const mgpu::range_t cta_seg_range { nv * cta, min(static_cast<int>(seg_count), nv * (cta + 1)) };
        const mgpu::range_t cta_vr_range { cta_seg_range.begin * 2, cta_seg_range.end * 2 };

        // const int max_cta_vr_count = 2 * nv;

        // __shared__ viewrayT shared_input[max_cta_vr_count];
        // __shared__ viewrayT shared_buffer[max_cta_vr_count];

        // __syncthreads();

        // Shared memory
        mgpu::range_t thr_seg_range { vt * tid, vt * (tid + 1) };
        mgpu::range_t thr_vr_range { thr_seg_range.begin * 2, thr_seg_range.end * 2 };

        // for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
        //     shared_input[i] = input[cta_vr_range.begin + i];

        // __syncthreads();

        viewrayT *shared_input = input + cta_vr_range.begin;
        viewrayT *shared_buffer = buffer + cta_vr_range.begin;

        viewrayT *shared_output = cta_visimergesort<launch_t>(shared_input, shared_buffer, thr_seg_range, thr_vr_range, tid);

        // for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
        //     input[cta_vr_range.begin + i] = shared_output[i];
    };

    if (profile) context.timer_begin();

    mgpu::cta_launch<launch_t>(k, num_ctas, context, dev_viewrays.data(), seg_count, dev_buffer.data());
    context.synchronize();

    if (profile)
    {
        double s = context.timer_end();
        std::cerr << "cta_visimergesort took " << s * 1e3 << "ms to process " << nv * num_ctas << " segments in "
                  << num_ctas << " blocks of " << nv << " segments (" << vr_count * sizeof(viewrayT) / double(1 << 30) / s
                  << " GiB/s)" << std::endl;
    }

    viewrayT *input_ptr = dev_viewrays.data();
    viewrayT *buffer_ptr = dev_buffer.data();

    auto comp_viewrayT = [] MGPU_DEVICE (const viewrayT & a, const viewrayT & b) {
        return a.t < b.t;
    };

    if (profile) context.timer_begin();

    for (int pass = 0; pass < num_passes; ++pass)
    {
        int coop = 2 << pass;

        mgpu::mem_t<int> partitions = mgpu::merge_sort_partitions(input_ptr, vr_count, coop, 2 * nv, comp_viewrayT, context);
        int *mp_data = partitions.data();

        auto k = [ = ] MGPU_DEVICE (const int tid, const int cta, viewray<T> *input, viewray<T> *output)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

            mgpu::range_t tile = mgpu::get_tile(cta, 2 * nv, vr_count);

            mgpu::merge_range_t cta_range = mgpu::compute_mergesort_range(vr_count, cta, coop, 2 * nv,
                                            mp_data[cta + 0], mp_data[cta + 1]);

            mgpu::merge_range_t merge_range = mgpu::compute_mergesort_range(vr_count, cta & ~(coop - 1), coop, 2 * nv);

            cta_visimerge(input, output, cta_range, merge_range, tile, tid, vt);
        };

        mgpu::cta_launch<launch_t>(k, num_ctas, context, input_ptr, buffer_ptr);
        context.synchronize();

        std::swap(input_ptr, buffer_ptr);
    }

    if (profile)
    {
        double s = context.timer_end();
        std::cerr << "it took " << s * 1e3 << "ms to process " << num_passes << " visimerge passes ("
                  << num_passes * vr_count * sizeof(viewrayT) / double(1 << 30) / s << " GiB/s)" << std::endl;
    }

    mgpu::dtoh(output, input_ptr, vr_count);
}


} // namespace vmgpu

#endif // VMGPU_KERNEL_VISIMERGESORT_H
