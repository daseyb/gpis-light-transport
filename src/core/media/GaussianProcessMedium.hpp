#ifndef GAUSSIANPROCESSMEDIUM_HPP_
#define GAUSSIANPROCESSMEDIUM_HPP_

#include "Medium.hpp"
#include "math/GaussianProcess.hpp"

namespace Tungsten {

class GaussianProcess;

enum class GPIntersectMethod {
    Mean,
    GPDiscrete
};

enum class GPNormalSamplingMethod {
    FiniteDifferences,
    ConditionedGaussian,
    Beckmann,
    GGX
};

struct GPContextFunctionSpace : public GPContext {
    std::vector<Vec3d> points;
    std::shared_ptr<GPRealNode> values;
    std::vector<Derivative> derivs;

    virtual void reset() override {
        points.clear();
        values.reset();
        derivs.clear();
    }
};

struct NormalDistribution {
    virtual bool isDeltaDistribution() = 0;
    virtual double evaluate(Vec3d normal) = 0;
    virtual Vec3d sample(PathSampleGenerator& sampler) = 0;
};

struct DeltaNormalDistribution : NormalDistribution {
    Vec3d normal;
    virtual bool isDeltaDistribution() override { return true; }
    virtual double evaluate(Vec3d normal) override { return 0.; }
    virtual Vec3d sample(PathSampleGenerator& sampler) override { return normal; }
};

struct MVNNormalDistribution : NormalDistribution {
    MultivariateNormalDistribution mvn;
    
    virtual bool isDeltaDistribution() override { return true; }
    virtual double evaluate(Vec3d normal) override { return 0.; }
    virtual Vec3d sample(PathSampleGenerator& sampler) override { return {}; }
};

class GaussianProcessMedium : public Medium
{
    Vec3f _materialSigmaA, _materialSigmaS;
    float _density;

    Vec3f _sigmaA, _sigmaS;
    Vec3f _sigmaT;
    bool _absorptionOnly;

    std::vector<std::shared_ptr<PhaseFunction>> _phaseFunctions;

protected:
    GPCorrelationContext _ctxt = GPCorrelationContext::Goldfish;
    GPIntersectMethod _intersectMethod = GPIntersectMethod::GPDiscrete;
    GPNormalSamplingMethod _normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian;

public:
    static GPCorrelationContext stringToCorrelationContext(const std::string& name);
    static std::string correlationContextToString(GPCorrelationContext ctxt);

    static GPIntersectMethod stringToIntersectMethod(const std::string& name);
    static std::string intersectMethodToString(GPIntersectMethod ctxt);

    static GPNormalSamplingMethod stringToNormalSamplingMethod(const std::string& name);
    static std::string normalSamplingMethodToString(GPNormalSamplingMethod ctxt);

    std::shared_ptr<GPSampleNode> _gp;
    GaussianProcessMedium();
    GaussianProcessMedium(std::shared_ptr<GPSampleNode> gp, std::vector<std::shared_ptr<PhaseFunction>> phases,
        float materialSigmaA, float materialSigmaS, float density,
        GPCorrelationContext ctxt = GPCorrelationContext::Goldfish, GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete, GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian) :
        _gp(gp), _phaseFunctions(phases), _materialSigmaA(materialSigmaA), _materialSigmaS(materialSigmaS), _density(density),
        _ctxt(ctxt), _intersectMethod(intersectMethod), _normalSamplingMethod(normalSamplingMethod)
    {}

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;
    virtual void loadResources() override;

    virtual bool isHomogeneous() const override;

    virtual void prepareForRender() override;

    virtual Vec3f sigmaA(Vec3f p) const override;
    virtual Vec3f sigmaS(Vec3f p) const override;
    virtual Vec3f sigmaT(Vec3f p) const override;

    virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const = 0;

    virtual std::shared_ptr<NormalDistribution> normalDistribution(const Ray& ray, const Vec3d& ip, MediumState& state) const { return nullptr; }

    bool intersect(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;
    virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const = 0;
    bool intersectMean(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;

    virtual bool sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
            MediumState &state, MediumSample &sample) const override;
    virtual Vec3f transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface, MediumSample* sample) const override;
    virtual float pdf(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const override;

    Vec3f sigmaA() const { return _sigmaA; }
    Vec3f sigmaS() const { return _sigmaS; }
};

}

#endif /* GAUSSIANPROCESSMEDIUM_HPP_ */
