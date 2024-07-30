#ifndef GRID_HPP_
#define GRID_HPP_

#include "math/Mat4f.hpp"
#include "math/Box.hpp"

#include "io/JsonSerializable.hpp"

namespace Tungsten {

enum class InterpolateMethod
{
    Point,
    Linear,
    Quadratic,
};

std::string interpolateMethodToString(InterpolateMethod method);
InterpolateMethod stringToInterpolateMethod(const std::string& name);

class PathSampleGenerator;

class Grid : public JsonSerializable
{
public:
    virtual ~Grid() {}

    virtual Mat4f naturalTransform() const;
    virtual Mat4f invNaturalTransform() const;
    virtual Box3f bounds() const;

    virtual void requestGradient() = 0;
    virtual void requestSDF() = 0;

    virtual float density(Vec3f p) const = 0;
    virtual Vec3f gradient(Vec3f p) const = 0;
    virtual Vec3f emission(Vec3f p) const = 0;
    virtual float opticalDepth(PathSampleGenerator &sampler, Vec3f p, Vec3f w, float t0, float t1) const = 0;
    virtual Vec2f inverseOpticalDepth(PathSampleGenerator &sampler, Vec3f p, Vec3f w, float t0, float t1, float xi) const = 0;
};

}

#endif /* GRID_HPP_ */
