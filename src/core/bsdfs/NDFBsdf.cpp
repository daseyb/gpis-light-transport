#include "NDFBsdf.hpp"
#include "ComplexIor.hpp"
#include "Fresnel.hpp"

#include "samplerecords/SurfaceScatterEvent.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "math/MathUtil.hpp"
#include "math/Angle.hpp"
#include "math/Vec.hpp"

#include "io/JsonObject.hpp"

#include <tinyformat/tinyformat.hpp>
#include <rapidjson/document.h>

#include <bsdfs/NDFs/GGX.h>
#include <bsdfs/lambert.h>


namespace Tungsten {

NDFBsdf::NDFBsdf()
: _materialName("Cu"), 
    _ndfType("beckmann"),
    _brdfType("conductor"),
  _eta(0.200438f, 0.924033f, 1.10221f),
  _k(3.91295f, 2.45285f, 2.14219f),
  micro_brdf(std::make_shared<ConductorBRDF>(1., 1.)), ndf(std::make_shared<BeckmannNDF>(micro_brdf.get(), 1., 1.)), macro_brdf(ndf.get())
{
    _lobes = BsdfLobes(BsdfLobes::SpecularReflectionLobe);
}

bool NDFBsdf::lookupMaterial()
{
    return ComplexIorList::lookup(_materialName, _eta, _k);
}

void NDFBsdf::fromJson(JsonPtr value, const Scene &scene)
{
    Bsdf::fromJson(value, scene);
    if (value.getField("eta", _eta) && value.getField("k", _k))
        _materialName.clear();
    if (value.getField("material", _materialName) && !lookupMaterial())
        value.parseError(tfm::format("Unable to find material with name '%s'", _materialName));

    value.getField("ndf", _ndfType);
    value.getField("brdf", _brdfType);
    value.getField("roughness", _roughness);
    value.getField("max_walk_length", _maxWalkLength);

    if (_brdfType == "conductor") {
        micro_brdf = std::make_shared<ConductorBRDF>(_eta.x(), _k.x());
        _lobes = BsdfLobes(BsdfLobes::SpecularReflectionLobe);
    }
    else {
        micro_brdf = std::make_shared<LambertBRDF>(1.);
        _lobes = BsdfLobes(BsdfLobes::DiffuseReflectionLobe);
    }

    if(_ndfType == "beckmann") {
        ndf = std::make_shared<BeckmannNDF>(micro_brdf.get(), _roughness.x(), _roughness.y()); // and assign it to microfacets with a GGX distribution
    } else if(_ndfType == "ggx") {
        ndf = std::make_shared<GGXNDF>(micro_brdf.get(), _roughness.x(), _roughness.y()); // and assign it to microfacets with a GGX distribution
    } else {
        ndf = std::make_shared<BeckmannNDF>(micro_brdf.get(), _roughness.x(), _roughness.y()); // and assign it to microfacets with a GGX distribution
    }

    macro_brdf = Microsurface(ndf.get(), _maxWalkLength);
}

rapidjson::Value NDFBsdf::toJson(Allocator &allocator) const
{
    JsonObject result{Bsdf::toJson(allocator), allocator,
        "type", "ndf",
        "roughness", _roughness,
        "ndf", _ndfType,
        "brdf", _brdfType
    };

    if (_materialName.empty())
        result.add("eta", _eta, "k", _k);
    else
        result.add("material", _materialName);

    return result;
}

bool NDFBsdf::sample(SurfaceScatterEvent &event) const
{
    double weight = 1.;
    event.wo = vec_conv<Vec3f>(macro_brdf.sample(1., 1., vec_conv<Vector3>(event.wi), weight));
    event.weight = Vec3f((float)weight);
    event.pdf = 1.f;
    return true;
}

Vec3f NDFBsdf::eval(const SurfaceScatterEvent &event) const
{
    return Vec3f((float)macro_brdf.eval(1., 1., vec_conv<Vector3>(event.wi), vec_conv<Vector3>(event.wo)));
}

bool NDFBsdf::invert(WritablePathSampleGenerator &/*sampler*/, const SurfaceScatterEvent &event) const
{
    return false;
}

float NDFBsdf::pdf(const SurfaceScatterEvent &event) const
{
    return 1.0f;
}

}
