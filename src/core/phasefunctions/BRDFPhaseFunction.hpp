#ifndef BRDFPHASEFUNCTION_HPP_
#define BRDFPHASEFUNCTION_HPP_

#include "PhaseFunction.hpp"
#include "core/bsdfs/Bsdf.hpp"

namespace Tungsten {

class BRDFPhaseFunction : public PhaseFunction
{
private:
    std::shared_ptr<Bsdf> _bsdf;

public:
    virtual rapidjson::Value toJson(Allocator& allocator) const override;
    virtual void fromJson(JsonPtr value, const Scene& scene) override;

    virtual Vec3f eval(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const override;
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
};

}

#endif /* BRDFPHASEFUNCTION_HPP_ */
