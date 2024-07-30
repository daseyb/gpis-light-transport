#include "BRDFPhaseFunction.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

void BRDFPhaseFunction::fromJson(JsonPtr value, const Scene& scene) {
    PhaseFunction::fromJson(value, scene);

    if (auto bsdf = value["bsdf"])
        _bsdf = scene.fetchBsdf(bsdf);
}


rapidjson::Value BRDFPhaseFunction::toJson(Allocator &allocator) const
{
    return JsonObject{PhaseFunction::toJson(allocator), allocator,
        "type", "brdf",
        "bsdf", * _bsdf,
    };
}

Vec3f BRDFPhaseFunction::eval(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const
{
    auto normal = vec_conv<Vec3f>(mediumSample.aniso.normalized());
    SurfaceScatterEvent se;
    se.requestedLobe = BsdfLobes::AllLobes;
    se.frame = TangentFrame(normal);
    se.wi = se.frame.toLocal(-wi).normalized();
    se.wo = se.frame.toLocal(wo).normalized();
    IntersectionInfo info;
    info.bsdf = _bsdf.get();
    info.Ng = normal;
    info.Ns = normal;
    info.p = Vec3f(0.0f);
    info.primitive = nullptr;
    info.uv = Vec2f(0.f);
    info.w = Vec3f(0.0f);
    se.info = &info;

    return _bsdf->eval(se);
}

bool BRDFPhaseFunction::sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const
{
    auto normal = vec_conv<Vec3f>(mediumSample.aniso.normalized());
    SurfaceScatterEvent se;
    se.sampler = &sampler;
    se.frame = TangentFrame(normal);
    se.wi = se.frame.toLocal(-wi).normalized();
    se.requestedLobe = BsdfLobes::AllLobes;
    IntersectionInfo info;
    info.bsdf = _bsdf.get();
    info.Ng = normal;
    info.Ns = normal;
    info.p = Vec3f(0.0f);
    info.primitive = nullptr;
    info.uv = Vec2f(0.f);
    info.w = Vec3f(0.0f);
    se.info = &info;

    if (!_bsdf->sample(se)) return false;
    sample.w = se.frame.toGlobal(se.wo).normalized();
    sample.weight = se.weight;
    sample.pdf = se.pdf;
    return true;
}

bool BRDFPhaseFunction::invert(WritablePathSampleGenerator &sampler, const Vec3f &/*wi*/, const Vec3f &wo, const MediumSample& mediumSample) const
{
    return false;
}

float BRDFPhaseFunction::pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const
{
    auto normal = vec_conv<Vec3f>(mediumSample.aniso.normalized());
    SurfaceScatterEvent se;
    se.frame = TangentFrame(normal);
    se.wi = se.frame.toLocal(-wi);
    se.wo = se.frame.toLocal(wo);
    se.requestedLobe = BsdfLobes::AllLobes;
    IntersectionInfo info;
    info.bsdf = _bsdf.get();
    info.Ng = normal;
    info.Ns = normal;
    info.p = Vec3f(0.0f);
    info.primitive = nullptr;
    info.uv = Vec2f(0.f);
    info.w = Vec3f(0.0f);
    se.info = &info;

    return _bsdf->pdf(se);
}

}
