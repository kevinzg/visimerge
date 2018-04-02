#ifndef VMGPU_SERIAL_VISIMERGESORT_H
#define VMGPU_SERIAL_VISIMERGESORT_H

#include <algorithm>
#include "serial_visimerge.h"

namespace vmgpu {


constexpr size_t log2(size_t x, size_t p = 0)
{
    return x > 1 ? log2(x / 2) + 1 : p;
}


template<typename T>
void init_viewrays(segment<T> *segs, size_t count, viewray<T> *dest)
{
    typedef viewray<T> viewrayT;
    typedef vec2<T> vec2T;

    for (size_t i = 0; i < count; ++i)
    {
        vec2T a = segs[i].a;
        vec2T b = segs[i].b;

        if (cross(a, b) < 0)
            std::swap(a, b);

        T atan2b = atan2(b);
        atan2b = atan2b == 0 ? 2 * M_PI : atan2b;

        dest[2 * i] = viewrayT(atan2(a), normalize(a), norm(a), CUDART_INF);
        dest[2 * i + 1] = viewrayT(atan2b, normalize(b), CUDART_INF, norm(b));
    }
}


template<typename T>
void serial_visimergesort(segment<T> *segs, size_t count, viewray<T> *output)
{
    typedef viewray<T> viewrayT;

    size_t num_passes = log2(count);

    viewrayT *input = output;
    viewrayT *buffer = new viewrayT[count * 2];

    if (num_passes & 1) std::swap(input, buffer);

    init_viewrays(segs, count, input);

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

#endif // VMGPU_SERIAL_VISIMERGESORT_H
