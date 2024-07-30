#ifndef TANGENTSPACE_HPP_
#define TANGENTSPACE_HPP_

#include <cmath>

#include "Mat4f.hpp"
#include "Vec.hpp"

namespace Tungsten {

template<typename Mat, typename Vec>
struct TangentFrameD
{
    Vec normal, tangent, bitangent;

    TangentFrameD() = default;

    TangentFrameD(const Vec& n, const Vec& t, const Vec& b)
        : normal(n), tangent(t), bitangent(b)
    {
    }

    template<typename ElT>
    auto signf(ElT v) {
        return v < 0 ? -1. : 1.;
    }

    TangentFrameD(const Vec& n)
        : normal(n)
    {
        // [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
        auto sign = signf(normal.z());
        auto a = -1.0f / (sign + normal.z());
        auto b = normal.x() * normal.y() * a;
        tangent = Vec(1.0 + sign * normal.x() * normal.x() * a, sign * b, -sign * normal.x());
        bitangent = Vec(b, sign + normal.y() * normal.y() * a, -normal.y());
    }

    Vec toLocal(const Vec& p) const
    {
        return Vec(
            tangent.dot(p),
            bitangent.dot(p),
            normal.dot(p)
        );
    }

    Vec toGlobal(const Vec& p) const
    {
        return tangent * p.x() + bitangent * p.y() + normal * p.z();
    }

    Mat toMatrix() const
    {
        Mat vmat;
        vmat.col(0) = tangent;
        vmat.col(1) = bitangent;
        vmat.col(2) = normal;
        return vmat;
    }
};

struct TangentFrame
{
    Vec3f normal, tangent, bitangent;

    TangentFrame() = default;

    TangentFrame(const Vec3f &n, const Vec3f &t, const Vec3f &b)
    : normal(n), tangent(t), bitangent(b)
    {
    }

    TangentFrame(const Vec3f &n)
    : normal(n)
    {
        // [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
        float sign = copysignf(1.0f, normal.z());
        const float a = -1.0f/(sign + normal.z());
        const float b = normal.x()*normal.y()*a;
        tangent = Vec3f(1.0f + sign*normal.x()*normal.x()*a, sign*b, -sign*normal.x());
        bitangent = Vec3f(b, sign + normal.y()*normal.y()*a, -normal.y());
    }

    Vec3f toLocal(const Vec3f &p) const
    {
        return Vec3f(
            tangent.dot(p),
            bitangent.dot(p),
            normal.dot(p)
        );
    }

    Vec3f toGlobal(const Vec3f &p) const
    {
        return tangent*p.x() + bitangent*p.y() + normal*p.z();
    }

    Mat4f toMatrix() const
    {
        return Mat4f(tangent, bitangent, normal);
    }
};


}

#endif /* TANGENTSPACE_HPP_ */
