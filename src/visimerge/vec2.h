#ifndef VMGPU_VEC2_H
#define VMGPU_VEC2_H

#include <moderngpu/meta.hxx>
#include "util.h"

namespace vmgpu {


template<typename T>
struct vec2
{
    T x, y;

    MGPU_HOST_DEVICE vec2(): vec2(0, 0) {}
    MGPU_HOST_DEVICE vec2(T x_, T y_): x(x_), y(y_) {}

    MGPU_HOST_DEVICE T operator[](size_t i) const { return i == 0 ? x : y; }
    MGPU_HOST_DEVICE T& operator[](size_t i) { return i == 0 ? x : y; }
};


template<typename T>
MGPU_HOST_DEVICE T cross(const vec2<T> &a, const vec2<T> &b)
{
    return a.x * b.y - a.y * b.x;
}


template<typename T>
MGPU_HOST_DEVICE vec2<T> operator+(const vec2<T> &a, const vec2<T> &b)
{
    return vec2<T>(a.x + b.x, a.y + b.y);
}


template<typename T>
MGPU_HOST_DEVICE vec2<T> operator-(const vec2<T> &a, const vec2<T> &b)
{
    return vec2<T>(a.x - b.x, a.y - b.y);
}


template<typename T, typename S>
MGPU_HOST_DEVICE vec2<T> operator*(const vec2<T> &a, const S &s)
{
    return vec2<T>(a.x * s, a.y * s);
}


template<typename T, typename S>
MGPU_HOST_DEVICE vec2<T> operator/(const vec2<T> &a, const S &s)
{
    return vec2<T>(a.x / s, a.y / s);
}


template<typename T>
MGPU_HOST_DEVICE T norm(const vec2<T> &a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}


template<typename T>
MGPU_HOST_DEVICE vec2<T> normalize(const vec2<T> &a)
{
    return a / norm(a);
}


template<typename T>
MGPU_HOST_DEVICE T atan2(const vec2<T> &a)
{
    T t = atan2(a.y, a.x);
    return t >= 0 ? t : t + 2 * M_PI;
}


template<typename T>
MGPU_HOST_DEVICE bool are_collinear(const vec2<T> &a, const vec2<T> &b, const vec2<T> &c)
{
    auto v = b - a;
    auto w = c - a;
    return almost_equal(cross(v, w), 0);
}


template<typename T>
MGPU_HOST_DEVICE bool almost_equal(const vec2<T> &a, const vec2<T> &b)
{
    return almost_equal(a.x, b.x) && almost_equal(a.y, b.y);
}


} // namespace vmgpu

#endif // VMGPU_VEC2_H
