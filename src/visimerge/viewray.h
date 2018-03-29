#ifndef VMGPU_VIEWRAY_H
#define VMGPU_VIEWRAY_H

#include <moderngpu/meta.hxx>

namespace vmgpu {


template<typename T>
struct viewray
{
    T t;
    vec2<T> v;
    T l, r;

    MGPU_HOST_DEVICE viewray(): viewray(0, vec2<T>(), 0, 0) {}
    MGPU_HOST_DEVICE viewray(T t_, vec2<T> v_, T l_, T r_): t(t_), v(v_), l(l_), r(r_) {}
};


} // namespace vmgpu

#endif // VMGPU_VIEWRAY_H
