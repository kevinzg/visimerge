#ifndef VMGPU_VIEWRAY_H
#define VMGPU_VIEWRAY_H

#include <cmath>
#include <moderngpu/meta.hxx>

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


} // namespace vmgpu

#endif // VMGPU_VIEWRAY_H
