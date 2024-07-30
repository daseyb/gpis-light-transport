#ifndef NDFBSDF_HPP_
#define NDFBSDF_HPP_

#include "Bsdf.hpp"
#include <bsdfs/conductor.h>
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/microsurface.h>

namespace Tungsten {

class Scene;

class NDFBsdf : public Bsdf
{
    std::string _materialName;
    std::string _ndfType;
    std::string _brdfType;
    Vec3f _eta;
    Vec3f _k;
    Vec2f _roughness;

    std::shared_ptr<BSDF> micro_brdf;
    std::shared_ptr<NDF> ndf;
    Microsurface macro_brdf;

    uint64 _maxWalkLength = MAX_WALK_LENGTH;

    bool lookupMaterial();

public:
    NDFBsdf();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool sample(SurfaceScatterEvent &event) const override;
    virtual Vec3f eval(const SurfaceScatterEvent &event) const override;
    virtual bool invert(WritablePathSampleGenerator &sampler, const SurfaceScatterEvent &event) const override;
    virtual float pdf(const SurfaceScatterEvent &event) const override;

    Vec3f eta() const
    {
        return _eta;
    }

    Vec3f k() const
    {
        return _k;
    }

    const std::string &materialName() const
    {
        return _materialName;
    }

    void setEta(Vec3f eta)
    {
        _eta = eta;
    }

    void setK(Vec3f k)
    {
        _k = k;
    }

    void setMaterialName(const std::string &materialName)
    {
        const std::string &oldMaterial = _materialName;
        _materialName = materialName;
        if (!lookupMaterial())
            _materialName = oldMaterial;
    }
};

}




#endif /* NDFBSDF_HPP_ */
