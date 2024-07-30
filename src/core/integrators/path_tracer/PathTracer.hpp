#ifndef PATHTRACER_HPP_
#define PATHTRACER_HPP_

#include "PathTracerSettings.hpp"

#include "integrators/TraceBase.hpp"

namespace Tungsten {

class PathTracer : public TraceBase
{
    PathTracerSettings _settings;
    bool _trackOutputValues;

public:
    PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId);

    Vec3f traceSample(Vec2u pixel, PathSampleGenerator &sampler);

    std::function<bool(const MediumSample& sample, Ray r)> _firstMediumBounceCb;
    std::function<bool(const SurfaceScatterEvent& ev, Ray r)> _firstSurfaceBounceCb;
};

}

#endif /* PATHTRACER_HPP_ */
