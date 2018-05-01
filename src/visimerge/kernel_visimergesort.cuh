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
#include <sys/time.h>
#include <unistd.h>
#include "cta_visimerge.cuh"
#include "cta_visimergesort.cuh"
#include "vec2.h"

namespace vmgpu {


template<typename T>
void init_viewrays(const segment<T> *segments, const int seg_count, const vray_array<T> &dest,
                   mgpu::context_t &context, bool profile = false)
{
    auto k = [] MGPU_DEVICE (const int i, const segment<T> *segments, const vray_array<T> dest)
    {
        int first = cross(segments[i].a, segments[i].b) < 0;

        const auto &a = segments[i][first];
        const auto &b = segments[i][!first];

        T atan2b = atan2(b);
        atan2b = atan2b == 0 ? 2 * M_PI : atan2b;

        dest.t[2 * i] = atan2(a);
        dest.t[2 * i + 1] = atan2b;

        dest.l[2 * i] = norm(a);
        dest.l[2 * i + 1] = CUDART_INF;

        dest.r[2 * i] = CUDART_INF;
        dest.r[2 * i + 1] = norm(b);
    };

    if (profile) context.timer_begin();

    mgpu::transform(k, seg_count, context, segments, dest);
    context.synchronize();

    if (profile)
    {
        double s = context.timer_end();
        std::cerr << "init_viewrays: " << s * 1e3 << "ms" << std::endl;
    }
}


template<typename launch_t, typename T>
void cta_level_visimergesort(const vray_array<T> &input, const int seg_count,
                             mgpu::context_t &context, bool profile = false)
{
    const int nv = launch_t::nv(context);
    const int num_ctas = mgpu::div_up(seg_count, nv);

    auto k = [] MGPU_DEVICE (const int tid, const int cta, const vray_array<T> input, const int seg_count)
    {
        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

        // Global memory
        const mgpu::range_t cta_seg_range { nv * cta, min(seg_count, nv * (cta + 1)) };
        const mgpu::range_t cta_vr_range { cta_seg_range.begin * 2, cta_seg_range.end * 2 };

        const int max_cta_vr_count = 2 * nv;

        __shared__ T shared_input_t[max_cta_vr_count];
        __shared__ T shared_input_l[max_cta_vr_count];
        __shared__ T shared_input_r[max_cta_vr_count];

        __shared__ T shared_buffer_t[max_cta_vr_count];
        __shared__ T shared_buffer_l[max_cta_vr_count];
        __shared__ T shared_buffer_r[max_cta_vr_count];

        __syncthreads();

        vray_array<T> shared_input { shared_input_t, shared_input_l, shared_input_r };
        vray_array<T> shared_buffer { shared_buffer_t, shared_buffer_l, shared_buffer_r };

        // Shared memory
        mgpu::range_t thr_seg_range { vt * tid, vt * (tid + 1) };
        mgpu::range_t thr_vr_range { thr_seg_range.begin * 2, thr_seg_range.end * 2 };

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            shared_input.t[i] = input.t[cta_vr_range.begin + i];

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            shared_input.l[i] = input.l[cta_vr_range.begin + i];

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            shared_input.r[i] = input.r[cta_vr_range.begin + i];

        __syncthreads();

        bool swap_input = cta_visimergesort<launch_t>(shared_input, shared_buffer,
                          thr_seg_range, thr_vr_range, tid);

        const vray_array<T> &shared_output = swap_input ? shared_buffer : shared_input;

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            input.t[cta_vr_range.begin + i] = shared_output.t[i];

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            input.l[cta_vr_range.begin + i] = shared_output.l[i];

        for (int i = thr_vr_range.begin; i < thr_vr_range.end; ++i)
            input.r[cta_vr_range.begin + i] = shared_output.r[i];
    };

    if (profile) context.timer_begin();

    mgpu::cta_launch<launch_t>(k, num_ctas, context, input, seg_count);
    context.synchronize();

    if (profile)
    {
        double s = context.timer_end();
        std::cerr << "cta_level_visimergesort: " << s * 1e3 << "ms" << std::endl;
    }
}


template<typename launch_t, typename T>
bool kernel_visimerge(vray_array<T> input, vray_array<T> buffer, mgpu::context_t &context,
                      const int seg_count, const int vr_count, const int max_passes = -1, bool profile = false)
{
    const int nv = launch_t::nv(context);
    const int num_ctas = mgpu::div_up(seg_count, nv);
    const int num_passes = max_passes < 0 ?
                           mgpu::find_log2(num_ctas, true) : min(mgpu::find_log2(num_ctas, true), max_passes);

    if (profile) context.timer_begin();

    for (int pass = 0; pass < num_passes; ++pass)
    {
        int coop = 2 << pass;

        mgpu::mem_t<int> partitions = mgpu::merge_sort_partitions(input.t, vr_count, coop, 2 * nv,
                                      mgpu::less_t<T>(), context);
        int *mp_data = partitions.data();

        auto k = [] MGPU_DEVICE (const int tid, const int cta, const int vr_count, const int coop,
                                 const vray_array<T> input, const vray_array<T> output, const int *mp_data)
        {
            typedef typename launch_t::sm_ptx params_t;
            enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

            mgpu::range_t tile = mgpu::get_tile(cta, 2 * nv, vr_count);

            mgpu::merge_range_t cta_range = mgpu::compute_mergesort_range(vr_count, cta, coop, 2 * nv,
                                            mp_data[cta + 0], mp_data[cta + 1]);

            mgpu::merge_range_t merge_range = mgpu::compute_mergesort_range(vr_count, cta & ~(coop - 1), coop, 2 * nv);

            cta_visimerge(input, output, cta_range, merge_range, tile, tid, vt);
        };

        mgpu::cta_launch<launch_t>(k, num_ctas, context, vr_count, coop, input, buffer, mp_data);
        context.synchronize();

        std::swap(input, buffer);
    }

    if (profile)
    {
        double s = context.timer_end();
        std::cerr << "kernel_visimerge: " << s * 1e3 << "ms" << std::endl;
    }

    return num_passes & 1;
}


template<typename T, typename launch_t>
void kernel_visimergesort(const mgpu::mem_t<segment<T>> &segments, vray_array<T> &dest,
                          mgpu::context_t &context, bool profile = false)
{
    const int seg_count = segments.size();
    const int vr_count = 2 * seg_count;

    vray_array<T> buffer = vray_array<T>::create(vr_count, context);

    struct timeval start, end;

    if (profile) gettimeofday(&start, NULL);

    init_viewrays(segments.data(), seg_count, dest, context, profile);

    cta_level_visimergesort<launch_t>(dest, seg_count, context, profile);

    bool swap_input = kernel_visimerge<launch_t>(dest, buffer, context, seg_count, vr_count, -1, profile);

    if (profile)
    {
        gettimeofday(&end, NULL);

        long seconds  = end.tv_sec  - start.tv_sec;
        long useconds = end.tv_usec - start.tv_usec;

        double s = seconds + useconds / 1e6;

        std::cerr << "total: " << s * 1e3 << "ms" << std::endl;
    }

    if (swap_input)
        std::swap(buffer, dest);

    vray_array<T>::destroy(buffer, context);
}


} // namespace vmgpu

#endif // VMGPU_KERNEL_VISIMERGESORT_H
