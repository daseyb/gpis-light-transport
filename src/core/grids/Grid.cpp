#include "Grid.hpp"
#include <Debug.hpp>

namespace Tungsten {

std::string interpolateMethodToString(InterpolateMethod method)
{
    switch (method) {
    default:
    case InterpolateMethod::Point: return "point";
    case InterpolateMethod::Linear: return "linear";
    case InterpolateMethod::Quadratic:  return "quadratic";
    }
}


InterpolateMethod stringToInterpolateMethod(const std::string& name)
{
    if (name == "point")
        return InterpolateMethod::Point;
    else if (name == "linear")
        return InterpolateMethod::Linear;
    else if (name == "quadratic")
        return InterpolateMethod::Quadratic;
    FAIL("Invalid interpolate method: '%s'", name);
}

Mat4f Grid::naturalTransform() const
{
    return Mat4f();
}

Mat4f Grid::invNaturalTransform() const
{
    return Mat4f();
}

Box3f Grid::bounds() const
{
    return Box3f();
}

}
