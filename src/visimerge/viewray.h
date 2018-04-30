#ifndef VMGPU_VIEWRAY_H
#define VMGPU_VIEWRAY_H

#include <cmath>
#include <moderngpu/meta.hxx>

#ifdef __CUDACC__
#include <moderngpu/context.hxx>
#endif // __CUDACC__

using std::cos;
using std::sin;

namespace vmgpu {


template<typename T>
struct viewray
{
    T t;
#ifndef __CUDACC__
    vec2<T> v;
#endif // __CUDACC__
    T l, r;

#ifndef __CUDACC__
    MGPU_HOST_DEVICE viewray(): viewray(0, vec2<T>(), 0, 0) {}
    MGPU_HOST_DEVICE viewray(T t_, vec2<T> v_, T l_, T r_): t(t_), v(v_), l(l_), r(r_) {}

    MGPU_HOST_DEVICE T vx() const { return v.x; }
    MGPU_HOST_DEVICE T vy() const { return v.y; }
#else
    MGPU_HOST_DEVICE viewray(): viewray(0, 0, 0) {}
    MGPU_HOST_DEVICE viewray(T t_, T l_, T r_): t(t_), l(l_), r(r_) {}

    MGPU_HOST_DEVICE T vx() const { return cos(t); }
    MGPU_HOST_DEVICE T vy() const { return sin(t); }
#endif // __CUDACC__
};


#ifdef __CUDACC__
template<typename T>
struct MGPU_ALIGN(16) vray_array
{
    T *t{nullptr};
    T *l{nullptr};
    T *r{nullptr};

    static vray_array create(size_t vr_count, mgpu::context_t &context,
                             mgpu::memory_space_t space = mgpu::memory_space_device)
    {
        vray_array vr;
        vr.t = static_cast<T*>(context.alloc(sizeof(T) * vr_count, space));
        vr.l = static_cast<T*>(context.alloc(sizeof(T) * vr_count, space));
        vr.r = static_cast<T*>(context.alloc(sizeof(T) * vr_count, space));

        return vr;
    }

    static void destroy(vray_array &vr, mgpu::context_t &context,
                        mgpu::memory_space_t space = mgpu::memory_space_device)
    {
        if (vr.t) context.free(vr.t, space);
        if (vr.l) context.free(vr.l, space);
        if (vr.r) context.free(vr.r, space);

        vr.t = vr.l = vr.r = nullptr;
    }

    MGPU_HOST_DEVICE vray_array operator+(const int d) const
    {
        return vray_array { this->t + d, this->l + d, this->r + d };
    }
};
#endif // __CUDACC__


} // namespace vmgpu

#endif // VMGPU_VIEWRAY_H
