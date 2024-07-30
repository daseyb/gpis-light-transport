#ifndef ISOTROPICPHASEFUNCTION_HPP_
#define ISOTROPICPHASEFUNCTION_HPP_

#include "PhaseFunction.hpp"
#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

namespace Tungsten {

class IsotropicPhaseFunction : public PhaseFunction
{
public:
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual Vec3f eval(const Vec3f&/*wi*/, const Vec3f&/*wo*/, const MediumSample& /*mediumSample*/) const override
    {
        return Vec3f(INV_FOUR_PI);
    }
        
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &/*wi*/, const MediumSample& /*mediumSample*/, PhaseSample &sample) const override
    {
        sample.w = SampleWarp::uniformSphere(sampler.next2D());
        sample.weight = Vec3f(1.0f);
        sample.pdf = SampleWarp::uniformSpherePdf();
        return true;
    }
        
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &/*wi*/, const Vec3f &wo, const MediumSample& /*mediumSample*/) const
    {
        sampler.put2D(SampleWarp::invertUniformSphere(wo, sampler.untracked1D()));
        return true;
    }
        
    virtual float pdf(const Vec3f &/*wi*/, const Vec3f &/*wo*/, const MediumSample& /*mediumSample*/) const override
    {
        return SampleWarp::uniformSpherePdf();
    }

    /*virtual Vec3f eval(const Vec3f& wi, const Vec3f& wo, const MediumSample& mediumSample) const override;
    virtual bool sample(PathSampleGenerator& sampler, const Vec3f& wi, const MediumSample& mediumSample, PhaseSample& sample) const override;
    virtual bool invert(WritablePathSampleGenerator& sampler, const Vec3f& wi, const Vec3f& wo, const MediumSample& mediumSample) const;
    virtual float pdf(const Vec3f& wi, const Vec3f& wo, const MediumSample& mediumSample) const override;*/
};

}

#endif /* ISOTROPICPHASEFUNCTION_HPP_ */
