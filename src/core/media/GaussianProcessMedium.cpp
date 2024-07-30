#include "GaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

namespace Tungsten {

    std::string GaussianProcessMedium::correlationContextToString(GPCorrelationContext ctxt)
    {
        switch (ctxt) {
        default:
        case GPCorrelationContext::Elephant:  return "elephant";
        case GPCorrelationContext::Goldfish:   return "goldfish";
        case GPCorrelationContext::Dori:   return "dori";
        case GPCorrelationContext::None:   return "none";
        }
    }

    GPCorrelationContext GaussianProcessMedium::stringToCorrelationContext(const std::string& name)
    {
        if (name == "elephant")
            return GPCorrelationContext::Elephant;
        else if (name == "goldfish")
            return GPCorrelationContext::Goldfish;
        else if (name == "dori")
            return GPCorrelationContext::Dori;
        else if (name == "none")
            return GPCorrelationContext::None;
        FAIL("Invalid correlation context: '%s'", name);
    }

    std::string GaussianProcessMedium::intersectMethodToString(GPIntersectMethod ctxt)
    {
        switch (ctxt) {
        default:
        case GPIntersectMethod::Mean:       return "mean";
        case GPIntersectMethod::GPDiscrete: return "gp_discrete";
        }
    }

    GPIntersectMethod GaussianProcessMedium::stringToIntersectMethod(const std::string& name)
    {
        if (name == "mean")
            return GPIntersectMethod::Mean;
        else if (name == "gp_discrete")
            return GPIntersectMethod::GPDiscrete;
        FAIL("Invalid intersect method: '%s'", name);
    }

    std::string GaussianProcessMedium::normalSamplingMethodToString(GPNormalSamplingMethod val)
    {
        switch (val) {
        default:
        case GPNormalSamplingMethod::FiniteDifferences:  return "finite_differences";
        case GPNormalSamplingMethod::ConditionedGaussian:   return "conditioned_gaussian";
        case GPNormalSamplingMethod::Beckmann:   return "beckmann";
        case GPNormalSamplingMethod::GGX:   return "ggx";
        }
    }

    GPNormalSamplingMethod GaussianProcessMedium::stringToNormalSamplingMethod(const std::string& name)
    {
        if (name == "finite_differences")
            return GPNormalSamplingMethod::FiniteDifferences;
        else if (name == "conditioned_gaussian")
            return GPNormalSamplingMethod::ConditionedGaussian;
        else if (name == "beckmann")
            return GPNormalSamplingMethod::Beckmann;
        else if (name == "ggx")
            return GPNormalSamplingMethod::GGX;
        FAIL("Invalid normal sampling method: '%s'", name);
    }


    GaussianProcessMedium::GaussianProcessMedium()
        : _materialSigmaA(0.0f),
        _materialSigmaS(0.0f),
        _density(1.0f),
        _gp(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>())),
        _ctxt(GPCorrelationContext::Goldfish),
        _intersectMethod(GPIntersectMethod::GPDiscrete),
        _normalSamplingMethod(GPNormalSamplingMethod::ConditionedGaussian)
    {
    }

    void GaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        Medium::fromJson(value, scene);
        value.getField("sigma_a", _materialSigmaA);
        value.getField("sigma_s", _materialSigmaS);
        value.getField("density", _density);

        std::string ctxtString = "goldfish";
        value.getField("correlation_context", ctxtString);
        _ctxt = stringToCorrelationContext(ctxtString);

        std::string intersectString = "gp_discrete";
        value.getField("intersect_method", intersectString);
        _intersectMethod = stringToIntersectMethod(intersectString);

        std::string normalString = "conditioned_gaussian";
        value.getField("normal_method", normalString);
        _normalSamplingMethod = stringToNormalSamplingMethod(normalString);

        if (auto gp = value["gaussian_process"])
            _gp = scene.fetchGaussianProcess(gp);

        // We always have the default one
        _phaseFunctions.push_back(_phaseFunction);

        if (auto addPhaseFunctions = value["additional_phase_functions"]) {
            for (unsigned i = 0; i < addPhaseFunctions.size(); ++i)
                _phaseFunctions.emplace_back(scene.fetchPhase(addPhaseFunctions[i]));
        }
    }

    rapidjson::Value GaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ Medium::toJson(allocator), allocator,
            "type", "gaussian_process",
            "sigma_a", _materialSigmaA,
            "sigma_s", _materialSigmaS,
            "density", _density,
            "gaussian_process", *_gp,
            "correlation_context", correlationContextToString(_ctxt),
            "intersect_method", intersectMethodToString(_intersectMethod),
            "normal_method", normalSamplingMethodToString(_normalSamplingMethod),
        };
    }

    void GaussianProcessMedium::loadResources() {
        _gp->loadResources();
    }


    bool GaussianProcessMedium::isHomogeneous() const
    {
        return false;
    }

    void GaussianProcessMedium::prepareForRender()
    {
        _sigmaA = _materialSigmaA * _density;
        _sigmaS = _materialSigmaS * _density;
        _sigmaT = _sigmaA + _sigmaS;
        _absorptionOnly = _sigmaS == 0.0f;
    }

    Vec3f GaussianProcessMedium::sigmaA(Vec3f /*p*/) const
    {
        return _sigmaA;
    }

    Vec3f GaussianProcessMedium::sigmaS(Vec3f /*p*/) const
    {
        return _sigmaS;
    }

    Vec3f GaussianProcessMedium::sigmaT(Vec3f /*p*/) const
    {
        return _sigmaT;
    }

    bool GaussianProcessMedium::intersect(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        switch (_intersectMethod) {
        case GPIntersectMethod::Mean:
            return intersectMean(sampler, ray, state, t);
        case GPIntersectMethod::GPDiscrete:
            return intersectGP(sampler, ray, state, t);
        default:
            return false;
        }
    }

    bool GaussianProcessMedium::intersectMean(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        t = ray.nearT() + 0.0001f;
        auto deriv = Derivative::None;
        auto rp = vec_conv<Vec3d>(ray.pos());
        auto rd = vec_conv<Vec3d>(ray.dir());

        for(int i = 0; i < 2048*4; i++) {
            auto p = rp + t * rd;
            float m = _gp->mean(&p, &deriv, nullptr, Vec3d(0.f), 1)(0);

            if(m < 0.00001f) {
                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = { Derivative::None };
                ctxt->points = { p };
                //ctxt->values = { 0. };
                state.gpContext = ctxt;
                return true;
            }

            t += m * 0.4f;

            if(t >= ray.farT()) {
                return false;
            }
        }

        state.gpContext = std::make_shared<GPContextFunctionSpace>();
        std::cerr << "Ran out of iterations in mean intersect sphere trace." << std::endl;
        t = ray.farT();
        return false;
    }

    bool GaussianProcessMedium::sampleDistance(PathSampleGenerator & sampler, const Ray & ray,
        MediumState & state, MediumSample & sample) const
    {
        sample.emission = Vec3f(0.0f);
        auto r = ray;
        size_t matId = 0;

        double startT = r.nearT();
        if (!std::isfinite(r.farT())) {
            r.setFarT(startT + 2000);
        }

        float maxT = r.farT();

        if (state.bounce >= _maxBounce) {
            return false;
        }

        if (maxT == 0.f ) {
            sample.t = maxT;
            sample.weight = Vec3f(1.f);
            sample.pdf = 1.0f;
            sample.exited = true;
            sample.p = ray.pos() + sample.t * ray.dir();
            sample.phase = _phaseFunction.get();
            return true;
        }

        if (_absorptionOnly) {
            if (maxT == Ray::infinity())
                return false;
            sample.t = maxT;
            sample.weight = transmittance(sampler, ray, state.firstScatter, true, &sample);
            sample.pdf = 1.0f;
            sample.exited = true;
        }
        else {
            double t = maxT;

            auto ro = vec_conv<Vec3d>(r.pos());
            auto rd = vec_conv<Vec3d>(r.dir()).normalized();


            // Handle the "ray marching" case
            // I.e. we want to allow the intersect function to not handle the whole ray
            // In that case it will tell us it didn't intersect, but t will be less than ray.farT()
            do {
                r.setNearT((float)startT);
                sample.exited = !intersect(sampler, r, state, t);
                
                // We sample a gradient if:
                // (1) The ray did intersect a surface
                // (2) The ray did not intersect a surface, but we'll need to continue
                if (t < maxT) {

                    Vec3d ip = ro + rd * t;

                    Vec3d grad;
                    if (!sampleGradient(sampler, ray, ip, state, grad)) {
                        std::cout << "Failed to sample gradient.\n";
                        return false;
                    }

                    state.lastAniso = sample.aniso = grad;
                    state.firstScatter = false;

                    if (!std::isfinite(sample.aniso.avg())) {
                        sample.aniso = Vec3d(1.f, 0.f, 0.f);
                        std::cout << "Gradient invalid.\n";
                        return false;
                    }
                }

                startT = t;

                // We only keep going in the case where we haven't finished processing the ray yet.
            } while (t < maxT && sample.exited);

            if (!sample.exited) {
                if (sample.aniso.dot(vec_conv<Vec3d>(ray.dir())) > 0) {
                    //std::cout << "Sampled gradient at intersection point points in the wrong direction. "<< sample.aniso.dot(vec_conv<Vec3d>(ray.dir())) << "\n";
                    return false;
                }

                if (sample.aniso.lengthSq() < 0.0000001f) {
                    sample.aniso = Vec3d(1.f, 0.f, 0.f);
                    std::cout << "Gradient zero.\n";
                    return false;
                }

                sample.weight = sample.continuedWeight = vec_conv<Vec3f>(_gp->color(ro + rd * t));
                sample.emission = vec_conv<Vec3f>(_gp->emission(ro + rd * t));
            } else {
                sample.weight = sample.continuedWeight = Vec3f(1.f);
            }

            sample.t = min(float(t), maxT);
            sample.continuedT = float(t);
            sample.weight *= sigmaS(ray.pos() + sample.t * ray.dir()) / sigmaT(ray.pos() + sample.t * ray.dir());
            sample.continuedWeight *= sigmaS(ray.pos() + sample.continuedT * ray.dir()) / sigmaT(ray.pos() + sample.continuedT * ray.dir());
            sample.pdf = 1;

            state.lastAniso = sample.aniso;
            state.advance();
        }
        sample.p = ray.pos() + sample.t * ray.dir();

        sample.phase = _phaseFunctions[state.lastGPId].get();
        sample.gpId = state.lastGPId;
        sample.ctxt = state.gpContext.get();

        return true;
    }



    Vec3f GaussianProcessMedium::transmittance(PathSampleGenerator & sampler, const Ray & ray, bool startOnSurface,
        bool endOnSurface, MediumSample *sample) const
    {
        if (ray.farT() == Ray::infinity())
            return Vec3f(0.0f);

        MediumState state;
        state.reset();
        state.firstScatter = startOnSurface;
        
        if (sample) {
            state.lastAniso = sample->aniso;
            state.lastGPId = sample->gpId;
            // HACK: Non-owning shared_ptr
            state.gpContext = std::shared_ptr<GPContext>(sample->ctxt, [](GPContext*) {});
        }

        MediumSample smpl;
        if(!sampleDistance(sampler, ray, state, smpl)) {
            return Vec3f(0.f);
        }

        return smpl.exited ? smpl.weight : Vec3f(0.);
    }

    float GaussianProcessMedium::pdf(PathSampleGenerator&/*sampler*/, const Ray & ray, bool startOnSurface, bool endOnSurface) const
    {
        return 1.0f;
    }
}
