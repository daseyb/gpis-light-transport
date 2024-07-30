#ifndef GPPATHTRACER_HPP_
#define GPPATHTRACER_HPP_

#include "GPPathTracerSettings.hpp"

#include "integrators/TraceBase.hpp"

namespace Tungsten {

class GPPathTracer : public TraceBase
{
    GPPathTracerSettings _settings;
    bool _trackOutputValues;

public:
    GPPathTracer(TraceableScene *scene, const GPPathTracerSettings &settings, uint32 threadId);

    Vec3f traceSample(Vec2u pixel, PathSampleGenerator &sampler);
};

}

#endif /* GPPATHTRACER_HPP_ */
