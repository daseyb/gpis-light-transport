#include "TabulatedTransmittance.hpp"

#include "sampling/UniformPathSampler.hpp"

#include "math/MathUtil.hpp"

#include "io/JsonObject.hpp"

#include "Memory.hpp"

namespace Tungsten {

TabulatedTransmittance::TabulatedTransmittance()
{
}

void TabulatedTransmittance::fromJson(JsonPtr value, const Scene &scene)
{
    Transmittance::fromJson(value, scene);
    value.getField("ts", _ts);
    value.getField("pp", _pp);
    value.getField("pf", _pf);
    value.getField("fp", _fp);
    value.getField("ff", _ff);

    _pp_dist = std::make_unique<Distribution1D>(_pp);
    _fp_dist = std::make_unique<Distribution1D>(_fp);
}

rapidjson::Value TabulatedTransmittance::toJson(Allocator &allocator) const
{
    return JsonObject{Transmittance::toJson(allocator), allocator,
        "type", "tabulated",
        //"pp", _pp,
        //"pf", _pf,
        //"fp", _fp,
        //"ff", _ff,
    };
}

Vec3i lookupTabulationIdx(const Vec3f& tau, size_t tabCnt, float tabMax) {
    Vec3i requestedIdx = Vec3i(float(tabCnt) * tau / tabMax);
    if (requestedIdx[0] > tabCnt - 1 || requestedIdx[1] > tabCnt - 1 || requestedIdx[2] > tabCnt - 1) {
        //std::cout << "Requested index out of tabulated range: " << tau << " > " << tabMax << std::endl;
        requestedIdx = min(Vec3i(tabCnt - 1), requestedIdx);
    }
    return requestedIdx;
}

Vec3f TabulatedTransmittance::surfaceSurface(const Vec3f &tau) const
{
    Vec3i idx = lookupTabulationIdx(tau, _ts.size(), _ts.back());

    return {
        _ff[idx[0]],
        _ff[idx[1]],
        _ff[idx[2]],
    };
}

Vec3f TabulatedTransmittance::surfaceMedium(const Vec3f &tau) const
{
    Vec3i idx = lookupTabulationIdx(tau, _ts.size(), _ts.back());

    return {
        _fp[idx[0]],
        _fp[idx[1]],
        _fp[idx[2]],
    };
}
Vec3f TabulatedTransmittance::mediumSurface(const Vec3f &tau) const
{
    Vec3i idx = lookupTabulationIdx(tau, _ts.size(), _ts.back());

    return {
        _pf[idx[0]],
        _pf[idx[1]],
        _pf[idx[2]],
    };
}
Vec3f TabulatedTransmittance::mediumMedium(const Vec3f &tau) const
{
    Vec3i idx = lookupTabulationIdx(tau, _ts.size(), _ts.back());

    return {
        _pp[idx[0]],
        _pp[idx[1]],
        _pp[idx[2]],
    };
}

float TabulatedTransmittance::sigmaBar() const
{
    return _fp[0] / _pf[0];
}

bool TabulatedTransmittance::isDirac() const
{
    return false;
}

float TabulatedTransmittance::sampleSurface(PathSampleGenerator &sampler) const
{
    float u = sampler.next1D();
    int idx = 0;
    _fp_dist->warp(u, idx);
    return lerp(_ts[idx], _ts[idx + 1], u);
}
float TabulatedTransmittance::sampleMedium(PathSampleGenerator &sampler) const
{
    float u = sampler.next1D();
    int idx = 0;
    _pp_dist->warp(u, idx);
    return lerp(_ts[idx], _ts[idx + 1], u);
}

}
