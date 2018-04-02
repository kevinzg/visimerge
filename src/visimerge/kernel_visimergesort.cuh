#ifndef VMGPU_KERNEL_VISIMERGESORT_H
#define VMGPU_KERNEL_VISIMERGESORT_H

#include <iostream>
#include <algorithm>
#include <math_constants.h>
#include <moderngpu/transform.hxx>
#include <moderngpu/intrinsics.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/types.hxx>
#include "cta_visimerge.cuh"
#include "cta_visimergesort.cuh"
#include "vec2.h"

namespace vmgpu {


template<typename T>
void init_viewrays(segment<T> *segs, int count, viewray<T> *dest, mgpu::context_t &context)
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

        dest[2 * i] = viewrayT(atan2(a), normalize(a), norm(a), CUDART_INF);
        dest[2 * i + 1] = viewrayT(atan2b, normalize(b), CUDART_INF, norm(b));
    };

    mgpu::transform(k, count, context, segs, dest);

    context.synchronize();
}


template<typename T>
void kernel_visimergesort(segment<T> *segs, int seg_count, viewray<T> *output, mgpu::context_t &context)
{
    typedef segment<T> segmentT;
    typedef viewray<T> viewrayT;

    int vr_count = seg_count * 2;

    mgpu::mem_t<segmentT> dev_segments(seg_count, context);
    mgpu::htod(dev_segments.data(), segs, seg_count);

    mgpu::mem_t<viewrayT> dev_viewrays(vr_count, context);
    init_viewrays(dev_segments.data(), seg_count, dev_viewrays.data(), context);

    typedef mgpu::launch_params_t<16, 16> launch_t;

    int nv = launch_t::nv(context);
    int num_ctas = mgpu::div_up(seg_count, nv);
    int num_passes = mgpu::find_log2(num_ctas, true);

    auto k = [] MGPU_DEVICE (const int tid, const int cta, viewray<T> *input, int seg_count)
    {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        // Global memory
        const mgpu::range_t cta_seg_range { nv * cta, min(static_cast<int>(seg_count), nv * (cta + 1)) };
        const mgpu::range_t cta_vr_range { cta_seg_range.begin * 2, cta_seg_range.end * 2 };

        const int max_cta_vr_count = 2 * nv;

        __shared__ viewrayT shared_input[max_cta_vr_count];
        __shared__ viewrayT shared_buffer[max_cta_vr_count];

        __syncthreads();

        // Shared memory
        mgpu::range_t thr_seg_range { vt * tid, vt * (tid + 1) };
        mgpu::range_t thr_vr_range { thr_seg_range.begin * 2, thr_seg_range.end * 2 };

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            shared_input[i] = input[cta_vr_range.begin + i];

        __syncthreads();

        viewrayT *shared_output = cta_visimergesort<launch_t>(shared_input, shared_buffer, thr_seg_range, thr_vr_range, tid);

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            input[cta_vr_range.begin + i] = shared_output[i];
    };

    mgpu::cta_launch<launch_t>(k, num_ctas, context, dev_viewrays.data(), seg_count);

    context.synchronize();

    mgpu::dtoh(output, dev_viewrays.data(), vr_count);

    // The rest

    // size_t num_passes = mgpu::find_log2(count, true);

    // std::cerr << num_passes << std::endl;

    // viewrayT *input = output;
    // viewrayT *buffer = new viewrayT[vr_count];

    // if (num_passes & 1) std::swap(input, buffer);

    // mgpu::dtoh(input, dev_viewrays.data(), vr_count);

    // for (size_t p = 0; p < num_passes; ++p)
    // {
    //     size_t sub_count = 2 << p;

    //     for (size_t start = 0; start < 2 * count; start += 2 * sub_count)
    //     {
    //         viewrayT *a = input + start;
    //         viewrayT *b = a + sub_count;

    //         serial_visimerge(a, sub_count, b, sub_count, buffer + start);
    //     }

    //     std::swap(input, buffer);
    // }

    // delete[] buffer;
}


} // namespace vmgpu

#endif // VMGPU_KERNEL_VISIMERGESORT_H
