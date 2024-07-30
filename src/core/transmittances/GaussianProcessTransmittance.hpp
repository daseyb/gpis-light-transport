#ifndef GAUSSIANPROCESSTRANSMITTANCE_HPP_
#define GAUSSIANPROCESSTRANSMITTANCE_HPP_

#include "Transmittance.hpp"
#include "core/sampling/Distribution1D.hpp"

namespace Tungsten {

class GaussianProcessTransmittance : public Transmittance
{
    std::vector<float> _ts;
    std::vector<float> _pp;
    std::vector<float> _pf;
    std::vector<float> _fp;
    std::vector<float> _ff;

    std::unique_ptr<Distribution1D> _pp_dist, _fp_dist;


public:
    GaussianProcessTransmittance();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual Vec3f surfaceSurface(const Vec3f &tau) const override final;
    virtual Vec3f surfaceMedium(const Vec3f &tau) const override final;
    virtual Vec3f mediumSurface(const Vec3f &tau) const override final;
    virtual Vec3f mediumMedium(const Vec3f &tau) const override final;

    virtual bool isDirac() const override final;

    virtual float sigmaBar() const override final;

    virtual float sampleSurface(PathSampleGenerator &sampler) const override final;
    virtual float sampleMedium(PathSampleGenerator &sampler) const override final;
};

}

#endif /* GAUSSIANPROCESSTRANSMITTANCE_HPP_ */
