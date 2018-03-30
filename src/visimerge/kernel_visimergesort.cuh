#ifndef VMGPU_KERNEL_VISIMERGESORT_H
#define VMGPU_KERNEL_VISIMERGESORT_H

#include <iostream>
#include <algorithm>
#include <moderngpu/transform.hxx>
#include <moderngpu/intrinsics.hxx>
#include <moderngpu/memory.hxx>
#include "serial_visimerge.h"
#include "vec2.h"

namespace vmgpu {


template<typename T>
void init_viewrays(segment<T> *segs, size_t count, viewray<T> *dest, mgpu::context_t &context)
{
    typedef viewray<T> viewrayT;
    typedef vec2<T> vec2T;

    auto k = [ = ]MGPU_DEVICE(int i)
    {
        vec2T &a = segs[i].a;
        vec2T &b = segs[i].b;

        if (cross(a, b) < 0)
            mgpu::swap(a, b);

        T atan2b = atan2(b);
        atan2b = atan2b == 0 ? 2 * M_PI : atan2b;

        dest[2 * i] = viewrayT(atan2(a), normalize(a), norm(a), -1);
        dest[2 * i + 1] = viewrayT(atan2b, normalize(b), -1, norm(b));
    };

    mgpu::transform(k, count, context);

    context.synchronize();
}


template<typename T>
void kernel_visimergesort(segment<T> *segs, size_t count, viewray<T> *output, mgpu::context_t &context)
{
    typedef segment<T> segmentT;
    typedef viewray<T> viewrayT;

    mgpu::mem_t<segmentT> dev_segments(count, context);
    mgpu::htod(dev_segments.data(), segs, count);

    mgpu::mem_t<viewrayT> dev_viewrays(count * 2, context);

    init_viewrays(dev_segments.data(), count, dev_viewrays.data(), context);

    //

    size_t num_passes = mgpu::find_log2(count, true);

    std::cerr << num_passes << std::endl;

    viewrayT *input = output;
    viewrayT *buffer = new viewrayT[count * 2];

    if (num_passes & 1) std::swap(input, buffer);

    mgpu::dtoh(input, dev_viewrays.data(), count * 2);

    for (size_t p = 0; p < num_passes; ++p)
    {
        size_t sub_count = 2 << p;

        for (size_t start = 0; start < 2 * count; start += 2 * sub_count)
        {
            viewrayT *a = input + start;
            viewrayT *b = a + sub_count;

            serial_visimerge(a, sub_count, b, sub_count, buffer + start);
        }

        std::swap(input, buffer);
    }

    delete[] buffer;
}


} // namespace vmgpu

#endif // VMGPU_KERNEL_VISIMERGESORT_H
