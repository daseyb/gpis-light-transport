#include "SdfFunctions.hpp"
#include <math/Vec.hpp>
#include <math/Mat4f.hpp>
#include <math/MathUtil.hpp>
#include <math/Angle.hpp>
#include <Debug.hpp>

namespace Tungsten {

    std::string SdfFunctions::functionToString(SdfFunctions::Function val)
    {
        switch (val) {
        default:
        case Function::Knob:  return "knob";
        case Function::KnobInner:  return "knob_inner";
        case Function::KnobOuter:  return "knob_outer";
        case Function::TwoSpheres:  return "two_spheres";
        }
    }

    SdfFunctions::Function SdfFunctions::stringToFunction(const std::string& name)
    {
        if (name == "knob")
            return Function::Knob;
        else if (name == "knob_inner")
            return Function::KnobInner;
        else if (name == "knob_outer")
            return Function::KnobOuter;
        else if (name == "two_spheres")
            return Function::TwoSpheres;
        FAIL("Invalid sdf function: '%s'", name);
    }

    /*
    Copyright 2020 Towaki Takikawa @yongyuanxi
    The MIT License
    Link: N/A
    */

    /******************************************************************************
     * The MIT License (MIT)
     * Copyright (c) 2021, NVIDIA CORPORATION.
     * Permission is hereby granted, free of charge, to any person obtaining a copy of
     * this software and associated documentation files (the "Software"), to deal in
     * the Software without restriction, including without limitation the rights to
     * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
     * the Software, and to permit persons to whom the Software is furnished to do so,
     * subject to the following conditions:
     * The above copyright notice and this permission notice shall be included in all
     * copies or substantial portions of the Software.
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
     * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
     * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
     * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
     * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
     ******************************************************************************/

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // distance functions
    // taken from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float sdSphere(Vec3f v, float r) {
        return v.length() - r;
    }

    float sdTorus(Vec3f p, Vec2f t)
    {
        Vec2f q = Vec2f(p.xz().length() - t.x(), p.y());
        return q.length() - t.y();
    }

    float sdCone(Vec3f p, Vec2f c)
    {
        // c is the sin/cos of the angle
        float q = p.xy().length();
        return c.dot(Vec2f(q, p.z()));
    }

    float sdCappedCylinder(Vec3f p, float h, float r)
    {
        Vec2f d = abs(Vec2f(p.xz().length(), p.y())) - Vec2f(h, r);
        return min(max(d.x(), d.y()), 0.0f) + max(d, Vec2f(0.0f)).length();
    }

    float sdTriPrism(Vec3f p, Vec2f h)
    {
        Vec3f q = abs(p);
        return max(q.z() - h.y(), max(q.x() * 0.866025f + p.y() * 0.5f, -p.y()) - h.x() * 0.5f);
    }

    float opSmoothUnion(float d1, float d2, float k) {
        float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
        return lerp(d2, d1, h) - k * h * (1.0 - h);
    }
    float ssub(float d1, float d2, float k) {
        float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
        return lerp(d2, -d1, h) + k * h * (1.0 - h);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // actual distance functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float sdBase(Vec3f p) {
        // Intersect two cones
        float base = opSmoothUnion(sdCone(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), -90).transformVector(p + Vec3f(0.f, .9f, 0.f)),
            Vec2f(PI / 3., PI / 3.)),
            sdCone(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), 90).transformVector((p - Vec3f(0.f, .9f, 0.f))),
                Vec2f(PI / 3.f, PI / 3.f)),
            0.02);
        // Bound the base radius
        base = max(base, sdCappedCylinder(p, 1.1f, 0.25f)) * 0.7f;
        // Dig out the center
        base = max(-sdCappedCylinder(p, 0.6f, 0.3f), base);
        // Cut a slice of the pie
        base = max(-sdTriPrism(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), 90).transformVector(p + Vec3f(0.f, 0.f, -1.f)), Vec2f(1.2f, 0.3f)), base);
        return base;
    }

    float sdKnob(Vec3f p, int& mat) {
        float sphere = sdSphere(p, 1.0);
        float cutout = sdSphere(p - Vec3f(0.0f, 0.5f, 0.5f), 0.7);
        float cutout_etch = sdTorus(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), -45).transformVector((p - Vec3f(0.0f, 0.2f, 0.2f))), Vec2f(1.0f, 0.05f));
        float innersphere = sdSphere(p - Vec3f(0.0f, 0.0f, 0.0f), 0.75);

        // Cutout sphere
        float d = ssub(cutout, sphere, 0.1);

        // Add eye, etch the sphere
        d = min(d, innersphere);
        d = max(-cutout_etch, d);

        // Add base
        d = min(ssub(sphere,
            sdBase(p - Vec3f(0.f, -.775f, 0.f)), 0.1), d);
        return d;
    }


    float SdfFunctions::knob(Vec3f p, int& mat) {
        const float scale = 0.8;
        p *= 1. / scale;
        return sdKnob(p, mat) * scale;
    }

    float sdKnobInner(Vec3f p, int& mat) {
        return sdSphere(p - Vec3f(0.0f, 0.0f, 0.0f), 0.75);;
    }


    float SdfFunctions::knob_inner(Vec3f p, int& mat) {
        const float scale = 0.8;
        p *= 1. / scale;
        return sdKnobInner(p, mat) * scale;
    }

    float sdKnobOuter(Vec3f p, int& mat) {
        float sphere = sdSphere(p, 1.0);
        float cutout = sdSphere(p - Vec3f(0.0f, 0.5f, 0.5f), 0.7);
        float cutout_etch = sdTorus(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), -45).transformVector((p - Vec3f(0.0f, 0.2f, 0.2f))), Vec2f(1.0f, 0.05f));
        float innersphere = sdSphere(p - Vec3f(0.0f, 0.0f, 0.0f), 0.75);

        // Cutout sphere
        float d = ssub(cutout, sphere, 0.1);

        // Cut out eye, etch the sphere
        d = max(d, -innersphere);
        d = max(-cutout_etch, d);

        // Add base
        d = min(ssub(sphere,
            sdBase(p - Vec3f(0.f, -.775f, 0.f)), 0.1), d);
        return d;
    }

    float SdfFunctions::knob_outer(Vec3f p, int& mat) {
        const float scale = 0.8;
        p *= 1. / scale;
        return sdKnobOuter(p, mat) * scale;
    }

    float SdfFunctions::two_spheres(Vec3f p, int& mat) {
        return min((p - Vec3f(0., 10., 0.f)).length() - 9.5f, (p - Vec3f(0.f, -10.f, 0.f)).length() - 9.5f);
    }

    template<typename T>
    T fract(T v) {
        return v - floor(v);
    }

    template<typename Vec>
    auto dot(Vec a, Vec b) {
        return a.dot(b);
    }

    /* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
    Vec3f random3(Vec3f c) {
        float j = 4096.0 * sin(dot(c, Vec3f(17.0f, 59.4f, 15.0f)));
        Vec3f r;
        r.z() = fract(512.0 * j);
        j *= .125;
        r.x() = fract(512.0 * j);
        j *= .125;
        r.y() = fract(512.0 * j);
        return r - 0.5;
    }

    /* skew constants for 3d simplex functions */
    const float F3 = 0.3333333;
    const float G3 = 0.1666667;


    template<typename ElementType, unsigned Size>
    Tungsten::Vec<ElementType, Size> step(const Tungsten::Vec<ElementType, Size>& edge, const Tungsten::Vec<ElementType, Size>& x)
    {
        Tungsten::Vec<ElementType, Size> result;
        for (unsigned i = 0; i < Size; ++i)
            result[i] = x[i] < edge[i] ? 0 : 1;
        return result;
    }


    /* 3d simplex noise */
    float simplex3d(Vec3f p) {
        /* 1. find current tetrahedron T and it's four vertices */
        /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
        /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/

        /* calculate s and x */
        Vec3f s = std::floor(p + dot(p, Vec3f(F3)));
        Vec3f x = p - s + dot(s, Vec3f(G3));

        /* calculate i1 and i2 */
        Vec3f e = step(Vec3f(0.0), x - Vec3f(x.y(), x.z(), x.x()));
        Vec3f i1 = e * (1.0f - Vec3f(e.z(), e.x(), e.y()));
        Vec3f i2 = 1.0f - Vec3f(e.z(), e.x(), e.y()) * (1.0f - e);

        /* x1, x2, x3 */
        Vec3f x1 = x - i1 + G3;
        Vec3f x2 = x - i2 + 2.0f * G3;
        Vec3f x3 = x - 1.0f + 3.0f * G3;

        /* 2. find four surflets and store them in d */
        Vec4f w, d;

        /* calculate surflet weights */
        w.x() = dot(x, x);
        w.y() = dot(x1, x1);
        w.z() = dot(x2, x2);
        w.w() = dot(x3, x3);

        /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
        w = max(0.6f - w, Vec4f(0.0f));

        /* calculate surflet components */
        d.x() = dot(random3(s), x);
        d.y() = dot(random3(s + i1), x1);
        d.z() = dot(random3(s + i2), x2);
        d.w() = dot(random3(s + 1.0), x3);

        /* multiply d by w^4 */
        w *= w;
        w *= w;
        d *= w;

        /* 3. return the sum of the four surflets */
        return dot(d, Vec4f(52.0f));
    }

    // fbm function by https://code.google.com/p/fractalterraingeneration/wiki/Fractional_Brownian_Motion
    double fbm(Vec3d uv, int octaves)
    {
        float gain = 0.65;
        float lacunarity = 2.1042;

        float total = 0.0;
        float frequency = 0.5;
        float amplitude = gain;

        uv = uv * 5.0;

        total = simplex3d(Vec3f(uv));

        for (int i = 0; i < octaves; i++)
        {
            total += simplex3d(Vec3f(uv) * frequency) * amplitude;
            frequency *= lacunarity;
            amplitude *= gain;
        }

        total = (total + 2.0) / 4.0;

        return total;
    }


}