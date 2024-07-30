#include "WeightSpaceGaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformPathSampler.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

namespace Tungsten {


    WeightSpaceGaussianProcessMedium::WeightSpaceGaussianProcessMedium()
        : GaussianProcessMedium(
            std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>()),
            {},
            0.0f, 0.0f, 1.f),
        _numBasisFunctions(300),
        _useSingleRealization(false),
        _rayMarchStepSize(0.f)
    {
    }

    void WeightSpaceGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("basis_functions", _numBasisFunctions);
        value.getField("single_realization", _useSingleRealization);
        value.getField("step_size", _rayMarchStepSize);

        if (_useSingleRealization) {
            auto basisSampler = UniformPathSampler(0xdeadbeef);
            auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
            WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, _numBasisFunctions, basisSampler, Vec3d(0.), false, 3);

            _globalReal = WeightSpaceRealization::sample(std::make_shared<WeightSpaceBasis>(basis), gp, basisSampler);
        }
    }

    rapidjson::Value WeightSpaceGaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "weight_space_gaussian_process",
            "basis_functions", _numBasisFunctions,
            "single_realization", _useSingleRealization,
            "step_size", _rayMarchStepSize,
        };
    }

    bool WeightSpaceGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state, Vec3d& grad) const {

        GPContextWeightSpace& ctxt = *(GPContextWeightSpace*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch(_normalSamplingMethod) {
            case GPNormalSamplingMethod::FiniteDifferences:
            {
                float eps = 0.0001f;
                std::array<Vec3d, 6> gradPs{
                    ip + Vec3d(eps, 0.f, 0.f),
                    ip + Vec3d(0.f, eps, 0.f),
                    ip + Vec3d(0.f, 0.f, eps),
                    ip - Vec3d(eps, 0.f, 0.f),
                    ip - Vec3d(0.f, eps, 0.f),
                    ip - Vec3d(0.f, 0.f, eps),
                };

                std::array<double, 6> gradVs;
                for(int i = 0; i < 6; i++) {
                    gradVs[i] = ctxt.real.evaluate(gradPs[i]);
                }

                grad = Vec3d{
                    gradVs[0] - gradVs[3],
                    gradVs[1] - gradVs[4],
                    gradVs[2] - gradVs[5],
                } / (2 * eps);

                break;
            }
            case GPNormalSamplingMethod::ConditionedGaussian:
            {
                grad = ctxt.real.evaluateGradient(ip);
                break;
            }
            case GPNormalSamplingMethod::Beckmann:
            {
                auto deriv = Derivative::First;
                Vec3d normal = Vec3d(
                    _gp->mean(&ip, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                    _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                    _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

                TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

                Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
                float alpha = _gp->compute_beckmann_roughness(ip);
                BeckmannNDF ndf(0, alpha, alpha);

                grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
                break;
            }
            case GPNormalSamplingMethod::GGX:
            {
                auto deriv = Derivative::First;
                Vec3d normal = Vec3d(
                    _gp->mean(&ip, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                    _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                    _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

                TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

                Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
                float alpha = _gp->compute_beckmann_roughness(ip);
                GGXNDF ndf(0, alpha, alpha);
                grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
                break;
            }
        }


        return true;
    }

    bool WeightSpaceGaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        if(!state.gpContext) {
            auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
            auto ctxt = std::make_shared<GPContextWeightSpace>();

            if (_useSingleRealization) {
                ctxt->real = _globalReal;
            }
            else {
                WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, _numBasisFunctions, sampler, Vec3d(), false, 3);
                ctxt->real = WeightSpaceRealization::sample(std::make_shared<WeightSpaceBasis>(basis), gp, sampler);
            }
            
            state.gpContext = ctxt;
        }

        GPContextWeightSpace& ctxt = *(GPContextWeightSpace*)state.gpContext.get();
        const WeightSpaceRealization& real = ctxt.real;

        double farT = ray.farT();
        auto rd = vec_conv<Vec3d>(ray.dir());

        if (_rayMarchStepSize == 0) {
            const double sig_0 = (farT - ray.nearT()) * 0.1f;
            const double delta = 0.01;
            const double np = 1.5;
            const double nm = 0.5;

            t = 0;
            double sig = sig_0;

            auto p = vec_conv<Vec3d>(ray.pos()) + (t + ray.nearT()) * rd;
            double f0 = real.evaluate(p);

            int sign0 = f0 < 0 ? -1 : 1;

            for (int i = 0; i < 2048 * 4; i++) {
                auto p_c = p + (t + ray.nearT() + delta) * rd;
                double f_c = real.evaluate(p_c);
                int signc = f_c < 0 ? -1 : 1;

                if (signc != sign0) {
                    t += ray.nearT();
                    return true;
                }

                auto c = p + (t + ray.nearT() + sig * 0.5) * rd;
                auto v = sig * 0.5 * rd;

                double nsig;
                if (real.rangeBound(c, { v }) != RangeBound::Unknown) {
                    nsig = sig;
                    sig = sig * np;
                }
                else {
                    nsig = 0;
                    sig = sig * nm;
                }

                t += max(nsig, delta);

                if (t + ray.nearT() >= farT) {
                    t += ray.nearT();
                    return false;
                }
            }

            std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
            t = ray.farT();
            return false;
        }
        else {
            auto p = vec_conv<Vec3d>(ray.pos());
            double f0 = real.evaluate(p);
            int sign0 = f0 < 0 ? -1 : 1;

            double pf = f0;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            while (t < ray.farT()) {
                auto p_c = p + t * rd;
                double f_c = real.evaluate(p_c);
                int signc = f_c < 0 ? -1 : 1;
                if (signc != sign0) {
                    t = lerp(t - _rayMarchStepSize, t, (f_c - pf) / _rayMarchStepSize);
                    return true;
                }

                pf = f_c;
                t += _rayMarchStepSize;
            }

            t = ray.farT();
            return false;
        }
    }
}
