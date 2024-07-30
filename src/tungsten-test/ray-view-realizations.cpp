#include <core/math/GaussianProcess.hpp>
#include <core/media/FunctionSpaceGaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace Tungsten;


float rices_formula(float u, float L0, float L2) {
    return exp(-u * u / (2 * L0)) * sqrt(L2 / L0) / (2 * PI);
}

void sample_ray_realizations(std::shared_ptr<GaussianProcess> gp, int samples, Ray r, int numMicroSamplesPts, int numMacroSteps, GPCorrelationContext corrCtxt) {
    
    numMicroSamplesPts = (numMicroSamplesPts / numMacroSteps) * numMacroSteps;
    
    float meanv = gp->_mean->operator()(Derivative::None, Vec3d(0.f), Vec3d(1.f));
    FunctionSpaceGaussianProcessMedium gp_med(gp, 0, 1, 1, numMicroSamplesPts, corrCtxt);
    gp_med.prepareForRender();

    UniformPathSampler sampler(0);
    sampler.next2D();


    Path basePath = Path("testing/ray-realizations") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    std::string filename = basePath.asString() + tinyformat::format("/%s-%d-%d-%.2f-samples.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, numMacroSteps, r.farT());
    std::string filename_ts = basePath.asString() + tinyformat::format("/%s-%d-%d-%.2f-ts.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, numMacroSteps, r.farT());
    std::string filename_derivs = basePath.asString() + tinyformat::format("/%s-%d-%d-%.2f-derivs.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, numMacroSteps, r.farT() - r.nearT());

    std::cout << filename << "\n";

    float maxStepsize = (r.farT() - r.nearT()) / (numMicroSamplesPts * numMacroSteps);

    if (Path(filename).exists()) {
        //continue;
    }

    std::vector<double> sampledValuesAll;
    std::vector<double> sampledTsAll;
    std::vector<double> sampledDerivsAll;

    for (int s = 0; s < samples;) {

        std::vector<double> sampledValues;
        sampledValues.resize(numMicroSamplesPts * numMacroSteps);


        std::vector<double> sampledTs;
        sampledTs.resize(numMicroSamplesPts * numMacroSteps);

        std::vector<double> sampledDerivs;
        sampledDerivs.resize(numMacroSteps, 0.);

        Medium::MediumState state;
        state.reset();

        MediumSample sample;
        Ray tRay = r;
        tRay.setFarT(tRay.nearT() + maxStepsize * numMicroSamplesPts);
        bool success = gp_med.sampleDistance(sampler, tRay, state, sample);
        auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
        int macroStep = 0;

        {
            auto [values, ids] = ctxt->values->flatten();
            std::copy(values.begin(), values.end(), sampledValues.begin() + macroStep * numMicroSamplesPts);
        }

        std::transform(ctxt->points.begin(), ctxt->points.end(), sampledTs.begin() + macroStep * numMicroSamplesPts, [](auto pt) { return pt.x(); });
        sampledDerivs[macroStep] = sample.aniso.dot(vec_conv<Vec3d>(r.dir()));

        
        while (success && sample.exited && tRay.farT() < r.farT()) {
            tRay.setNearT(tRay.farT());
            tRay.setFarT(tRay.nearT() + maxStepsize * numMicroSamplesPts);
            success = gp_med.sampleDistance(sampler, tRay, state, sample);
            macroStep++;
            ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
            {
                auto [values, ids] = ctxt->values->flatten();
                std::copy(values.begin(), values.end(), sampledValues.begin() + macroStep * numMicroSamplesPts);
            }
            std::transform(ctxt->points.begin(), ctxt->points.end(), sampledTs.begin() + macroStep * numMicroSamplesPts, [](auto pt) { return pt.x(); });

            sampledDerivs[macroStep] = sample.aniso.dot(vec_conv<Vec3d>(r.dir()));
        }

        if (!success) {
            continue;
        }

        std::copy(sampledValues.begin(), sampledValues.end(), std::back_inserter(sampledValuesAll));
        std::copy(sampledTs.begin(), sampledTs.end(), std::back_inserter(sampledTsAll));
        std::copy(sampledDerivs.begin(), sampledDerivs.end(), std::back_inserter(sampledDerivsAll));
        s++;
    }

    {
        std::ofstream xfile(
            filename,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledValuesAll.data(), sizeof(double) * sampledValuesAll.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            filename_ts,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledTsAll.data(), sizeof(double) * sampledTsAll.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            filename_derivs,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledDerivsAll.data(), sizeof(double) * sampledDerivsAll.size());
        xfile.close();
    }
}


void ndf_cond_validate(std::shared_ptr<GaussianProcess> gp, int samples, std::string output,
    int numMicroSamplesPts, float maxStepsize, float zrange, GPCorrelationContext corrCtxt, float angle = (2 * PI) / 8,
    GPNormalSamplingMethod nsm = GPNormalSamplingMethod::ConditionedGaussian)
{

    Path basePath = Path("testing/ray-realizations") / Path(output) / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(
        gp, 0, 1, 1, numMicroSamplesPts,
        corrCtxt,
        GPIntersectMethod::GPDiscrete,
        nsm);
    gp_med->loadResources();
    gp_med->prepareForRender();

    UniformPathSampler sampler(0);
    sampler.next2D();

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle), 0.f, -cos(angle)));

    Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
    ray.setDir(mat.transformVector(ray.dir()).normalized());

    ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
    ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());

    if (maxStepsize == 0) {
        maxStepsize = (ray.farT() - ray.nearT()) / numMicroSamplesPts;
    }

    int numMacroSteps = int(std::ceil((ray.farT() - ray.nearT()) / (numMicroSamplesPts * maxStepsize)));

    std::string filename = basePath.asString() + tinyformat::format("/%s-%d-%d-%.2f-samples.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, numMacroSteps, ray.farT() - ray.nearT());
    std::string filename_ts = basePath.asString() + tinyformat::format("/%s-%d-%d-%.2f-ts.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, numMacroSteps, ray.farT() - ray.nearT());
    std::string filename_derivs = basePath.asString() + tinyformat::format("/%s-%d-%d-%.2f-derivs.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, numMacroSteps, ray.farT() - ray.nearT());

    std::cout << filename << "\n";
    std::cout << zrange << "\n";

    if (Path(filename).exists()) {
        //continue;
    }

    std::vector<double> sampledValuesAll;
    std::vector<double> sampledTsAll;
    std::vector<double> sampledDerivsAll;

    for (int s = 0; s < samples;) {
        std::vector<double> sampledValues;
        sampledValues.resize(numMicroSamplesPts * numMacroSteps, 0.);

        std::vector<double> sampledTs;
        sampledTs.resize(numMicroSamplesPts * numMacroSteps, ray.farT());

        std::vector<double> sampledDerivs;
        sampledDerivs.resize(numMacroSteps, 0.);

        Medium::MediumState state;
        state.reset();

        MediumSample sample;
        Ray tRay = ray;
        tRay.setFarT(tRay.nearT() + maxStepsize * numMicroSamplesPts);

        int macroStep = 0;

        bool success = gp_med->sampleDistance(sampler, tRay, state, sample);
        auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
        std::copy(ctxt->values.begin(), ctxt->values.begin() + min((size_t)numMicroSamplesPts, ctxt->values.size() - 1), sampledValues.begin() + macroStep * numMicroSamplesPts);
        std::transform(ctxt->points.begin(), ctxt->points.begin() + min((size_t)numMicroSamplesPts, ctxt->points.size() - 1), sampledTs.begin() + macroStep * numMicroSamplesPts, [rp = vec_conv<Vec3d>(ray.pos())](auto pt) { return (pt - rp).length(); });
        sampledDerivs[macroStep] = sample.aniso.dot(vec_conv<Vec3d>(ray.dir()));

        while (success && sample.exited && tRay.farT() < ray.farT()) {
            tRay.setNearT(tRay.farT());
            tRay.setFarT(tRay.nearT() + maxStepsize * numMicroSamplesPts);
            success = gp_med->sampleDistance(sampler, tRay, state, sample);
            macroStep++;
            ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
            std::copy(ctxt->values.begin(), ctxt->values.begin() + min((size_t)numMicroSamplesPts, ctxt->values.size() - 1), sampledValues.begin() + macroStep * numMicroSamplesPts);
            std::transform(ctxt->points.begin(), ctxt->points.begin() + min((size_t)numMicroSamplesPts, ctxt->points.size() - 1), sampledTs.begin() + macroStep * numMicroSamplesPts, [rp = vec_conv<Vec3d>(ray.pos())](auto pt) { return (pt - rp).length(); });
            sampledDerivs[macroStep] = sample.aniso.dot(vec_conv<Vec3d>(ray.dir()));
        }

        if (!success) {
            continue;
        }

        std::copy(sampledValues.begin(), sampledValues.end(), std::back_inserter(sampledValuesAll));
        std::copy(sampledTs.begin(), sampledTs.end(), std::back_inserter(sampledTsAll));
        std::copy(sampledDerivs.begin(), sampledDerivs.end(), std::back_inserter(sampledDerivsAll));
        s++;
    }

    {
        std::ofstream xfile(
            filename,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledValuesAll.data(), sizeof(double) * sampledValuesAll.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            filename_ts,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledTsAll.data(), sizeof(double) * sampledTsAll.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            filename_derivs,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledDerivsAll.data(), sizeof(double) * sampledDerivsAll.size());
        xfile.close();
    }
}


void ndf_cond_validate_new(std::shared_ptr<GaussianProcess> gp, int samples, std::string output,
    int numMicroSamplesPts, float zrange, GPCorrelationContext corrCtxt, float angle = (2 * PI) / 8,
    GPNormalSamplingMethod nsm = GPNormalSamplingMethod::ConditionedGaussian)
{

    Path basePath = Path("testing/ray-realizations/") / Path(output) / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(
        gp, 0, 1, 1, numMicroSamplesPts,
        corrCtxt,
        GPIntersectMethod::GPDiscrete,
        nsm);

    gp_med->loadResources();
    gp_med->prepareForRender();

    UniformPathSampler sampler(0);
    sampler.next2D();

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle), 0.f, -cos(angle)));

    Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
    ray.setDir(mat.transformVector(ray.dir()).normalized());

    ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
    ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());

    std::string filename = basePath.asString() + tinyformat::format("/%s-%d-%.2f-samples.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, ray.farT() - ray.nearT());
    std::string filename_ts = basePath.asString() + tinyformat::format("/%s-%d-%.2f-ts.bin", GaussianProcessMedium::correlationContextToString(corrCtxt), numMicroSamplesPts, ray.farT() - ray.nearT());

    std::cout << filename << "\n";

    if (Path(filename).exists()) {
        //continue;
    }

    std::vector<double> sampledValuesAll;
    std::vector<double> sampledTsAll;

    float percentCorrect = 0.f;

    for (int s = 0; s < samples;) {

        std::vector<double> sampledValues(numMicroSamplesPts * 2, 0.);
        std::vector<double> sampledTs(numMicroSamplesPts * 2, 10000.);

        Medium::MediumState state;
        state.reset();

        MediumSample sample;

        bool success = gp_med->sampleDistance(sampler, ray, state, sample);
        if (!success || sample.exited) {
            continue;
        }

        auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
        std::copy(ctxt->values.begin(), ctxt->values.begin()+ min((size_t)numMicroSamplesPts, ctxt->points.size()), sampledValues.begin());
        std::transform(ctxt->points.begin(), ctxt->points.begin() + min((size_t)numMicroSamplesPts, ctxt->points.size()), sampledTs.begin(), [rp = ctxt->points[0]](auto pt) { return (pt - rp).length(); });

        auto bounceRay = Ray(sample.p, -ray.dir());
        bounceRay.setNearT(0.f);
        bounceRay.setFarT(sample.t-ray.nearT());

        gp_med->sampleDistance(sampler, bounceRay, state, sample);
        ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);
        std::copy(ctxt->values.begin(), ctxt->values.begin() + min((size_t)numMicroSamplesPts, ctxt->points.size()), sampledValues.begin() + numMicroSamplesPts);
        std::transform(ctxt->points.begin(), ctxt->points.begin() + min((size_t)numMicroSamplesPts, ctxt->points.size()), sampledTs.begin() + numMicroSamplesPts, [rp = vec_conv<Vec3d>(bounceRay.pos()), off = bounceRay.farT()](auto pt) { return off + (pt - rp).length(); });


        if (sample.exited) {
            percentCorrect += 1;
        }
        
        if (!success) {
            continue;
        }
        
        std::copy(sampledValues.begin(), sampledValues.end(), std::back_inserter(sampledValuesAll));
        std::copy(sampledTs.begin(), sampledTs.end(), std::back_inserter(sampledTsAll));
        s++;
    }

    std::cout << "Percent correct: " << percentCorrect / samples << "\n";

    {
        std::ofstream xfile(
            filename,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledValuesAll.data(), sizeof(double) * sampledValuesAll.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            filename_ts,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledTsAll.data(), sizeof(double) * sampledTsAll.size());
        xfile.close();
    }

}


int main() {
    

    auto mean = std::make_shared<LinearMean>(Vec3d(0.), Vec3d(0., 0., 1.), 1.);
    //auto mean = std::make_shared<HomogeneousMean>(8.0f);
    auto gp = std::make_shared<GaussianProcess>(mean, std::make_shared<SquaredExponentialCovariance>(1.0f, 0.1f));
    //auto gp = std::make_shared<GaussianProcess>(mean, std::make_shared<SquaredExponentialCovariance>(2.0f, 1.0f));
    gp->_cov->_aniso = Vec3f(1.0f, 1.0f, 0.f);
    gp->_covEps = 0;

    auto stepSize = gp->goodStepsize(Vec3d(0.), 0.99);
    auto bound = gp->noIntersectBound(Vec3d(0.), 0.99999);

    int macroSteps = 4;
    int microsteps = 256;

    auto distance = stepSize * macroSteps * microsteps;

    Ray r(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.f, distance);

    //sample_ray_realizations(gp, 50, r, 256, 1, GPCorrelationContext::Goldfish);
    //sample_ray_realizations(gp, 50, r, 128, 4, GPCorrelationContext::Dori);
    //sample_ray_realizations(gp, 50, r, 128, 4, GPCorrelationContext::Goldfish);
    //sample_ray_realizations(gp, 50, r, microsteps, macroSteps, GPCorrelationContext::Goldfish);
    //sample_ray_realizations(gp, 50, r, 64, 4, GPCorrelationContext::Elephant);
    //sample_ray_realizations(gp, 50, r, 32, 4, GPCorrelationContext::Elephant);
    //sample_ray_realizations(gp, 50, r, 16, 4, GPCorrelationContext::Elephant);

    int samples = 500;

    //ndf_cond_validate(gp, samples, "microfacet", microsteps, stepSize, bound, GPCorrelationContext::Goldfish, 0. / 180 * PI);
    //ndf_cond_validate(gp, samples, "microfacet", microsteps, stepSize, bound, GPCorrelationContext::Goldfish, 10. / 180 * PI);
    
    ndf_cond_validate_new(gp, samples, "retro", 256, bound, GPCorrelationContext::Goldfish, 45. / 180 * PI);
    ndf_cond_validate_new(gp, samples, "retro", 256, bound, GPCorrelationContext::Dori, 45. / 180 * PI);
    ndf_cond_validate_new(gp, samples, "retro", 256, bound, GPCorrelationContext::Elephant, 45. / 180 * PI);
    
    //ndf_cond_validate(gp, samples, "microfacet", microsteps, stepSize, bound, GPCorrelationContext::Goldfish, 89. / 180 * PI);



    /*
    std::cout << "\npp" << std::endl;
    {
        std::vector<float> ts;
        for (int i = 0; i < sample_count; i++) {
            if (i % 100 == 0) {
                std::cout << i << "/" << sample_count << "\r";
            }

            Medium::MediumState state;
            state.reset();
            state.firstScatter = false;
            MediumSample sample;
            if (gp_med.sampleDistance(sampler, ray, state, sample)) {
                ts.push_back(sample.continuedT);
            }
        }

        {
            std::ofstream xfile(tinyformat::format("transmittance/%s-sample-pp.bin", gp->_cov->id()), std::ios::out | std::ios::binary);
            xfile.write((char*)ts.data(), sizeof(float) * ts.size());
            xfile.close();
        }
    }*/

}
