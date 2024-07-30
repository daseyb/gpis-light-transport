#ifndef LAMBERTIANPHASEFUNCTION_HPP_
#define LAMBERTIANPHASEFUNCTION_HPP_

#include "PhaseFunction.hpp"
#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"
#include "math/TangentFrame.hpp"

namespace Tungsten {

class LambertianPhaseFunction : public PhaseFunction
{
public:
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    float lambertianPdf(float cosTheta) const {
        float theta = acos(cosTheta);
        return 2*(sin(theta) - theta*cosTheta) / (3 * PI * PI);
    }

    virtual Vec3f eval(const Vec3f& wi, const Vec3f& wo, const MediumSample& /*mediumSample*/) const override
    {
        return Vec3f(lambertianPdf(wi.dot(wo)));
    }
        
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& /*mediumSample*/, PhaseSample &sample) const override
    {
        float zeta_1 = sampler.next1D();
        float zeta_2 = sampler.next1D();
        float zeta_3 = sampler.next1D();

        float mu = sqrt((1-zeta_1)*(1-zeta_2)) * sin(2*PI*zeta_3) - sqrt(zeta_1*zeta_2);
        float sinTheta = sin(acos(mu));
        float phi = sampler.next1D()*TWO_PI;

        sample.w = TangentFrame(wi).toGlobal(Vec3f(
            std::cos(phi)*sinTheta,
            std::sin(phi)*sinTheta,
            mu
        ));
        sample.weight = Vec3f(1.0f);
        sample.pdf = lambertianPdf(mu);
        return true;
    }
        
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &/*wi*/, const Vec3f &wo, const MediumSample& /*mediumSample*/) const
    {
        return false;
    }
        
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& /*mediumSample*/) const override
    {
        return lambertianPdf(wi.dot(wo));
    }
};

}

#endif /* LAMBERTIANPHASEFUNCTION_HPP_ */
