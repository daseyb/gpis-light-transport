#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/media/FunctionSpaceGaussianProcessMedium.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/ImageIO.hpp>
#include <io/FileUtils.hpp>
#include <bsdfs/Microfacet.hpp>
#include <rapidjson/document.h>
#include <io/JsonDocument.hpp>
#include <io/JsonObject.hpp>
#include <io/Scene.hpp>
#include <math/GaussianProcessFactory.hpp>
#include <io/MultiProgress.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

size_t gidx(int i, int j) {
    return i * NUM_SAMPLE_POINTS + j;
}

Eigen::MatrixXf compute_normals(const Eigen::MatrixXf& samples) {

    Eigen::MatrixXf normals(3, NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    normals.setZero();

    auto samplesr = samples.reshaped(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS);

    for (int i = 1; i < NUM_SAMPLE_POINTS-1; i++) {
        for (int j = 1; j < NUM_SAMPLE_POINTS-1; j++) {

            float eps = 2.f / NUM_SAMPLE_POINTS;
            
            auto r = samplesr(i + 1, j);
            auto l = samplesr(i - 1, j);
            auto b = samplesr(i, j + 1);
            auto t = samplesr(i, j - 1);

            Vec3f norm = Vec3f(-(r - l) / (2*eps), (b - t) / (2*eps), 1.f).normalized();

            normals(0, gidx(j, i)) = norm.x();
            normals(1, gidx(j, i)) = norm.y();
            normals(2, gidx(j, i)) = norm.z();
        }
    }

    return normals;
}

float compute_beckmann_roughness(const CovarianceFunction& cov) {
    float L2 = cov(Derivative::First, Derivative::First, Vec3d(0.f), Vec3d(0.f), Vec3d(1.f, 0.f, 0.f), Vec3d(1.f, 0.f, 0.f));
    return sqrt(2 * L2);
}


void normals_and_stuff(const GaussianProcess& gp, std::string output) {
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();


    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }


        auto [samplesD, ids] = gp.sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3d(1.0f, 0.0f, 0.0f), 1, sampler)->flatten();

        Eigen::MatrixXf samples = samplesD.cast<float>();

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;


        Eigen::MatrixXf thresholded(samples.rows(), samples.cols());
        {
            for (int i = 0; i < samples.cols(); i++) {
                for (int j = 0; j < samples.rows(); j++) {
                    thresholded(j, i) = samples(j, i) < 0 ? 0 : 1;
                }
            }
        }

        Path basePath = Path(output) / Path(gp._cov->id());

        if (!basePath.exists()) {
            FileUtils::createDirectory(basePath);
        }

        for (int i = 0; i < samples.cols(); i++) {
            Eigen::MatrixXf normals = compute_normals(samples.col(i));
            normals.array() += 1.0f;
            normals.array() *= 0.5f;

            samples.col(i).array() -= samples.col(i).minCoeff();
            samples.col(i).array() /= samples.col(i).maxCoeff();

            ImageIO::saveHdr(incrementalFilename(basePath / Path("rel.exr"), "", false), samples.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename(basePath / Path("thr.exr"), "", false), thresholded.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename(basePath / Path("normals.exr"), "", false), normals.data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 3);
        }
    }
}

void side_view(const GaussianProcess& gp, std::string output) {
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 20. * (Vec3d((float)j, 0.f, (float)i) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }


        auto [samplesD, ids] = gp.sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3d(1.0f, 0.0f, 0.0f), 1, sampler)->flatten();


        Eigen::MatrixXf samples = samplesD.cast<float>();

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;


        Eigen::MatrixXf thresholded(samples.rows(), samples.cols());
        {
            for (int i = 0; i < samples.cols(); i++) {
                for (int j = 0; j < samples.rows(); j++) {
                    thresholded(j, i) = samples(j, i) < 0 ? 0 : 1;
                }
            }
        }

        Path basePath = Path(output) / Path(gp._cov->id());
        if (!basePath.exists()) {
            FileUtils::createDirectory(basePath);
        }

        for (int i = 0; i < samples.cols(); i++) {

            std::ofstream xfile(incrementalFilename(basePath + Path("-rel.bin"), "", false).asString(), std::ios::out | std::ios::binary);
            xfile.write((char*)samples.col(i).data(), sizeof(float) * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
            xfile.close();

            samples.col(i).array() -= samples.col(i).minCoeff();
            samples.col(i).array() /= samples.col(i).maxCoeff();

            ImageIO::saveHdr(incrementalFilename(basePath + Path("-rel.exr"), "", false), samples.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename(basePath + Path("-thr.exr"), "", false), thresholded.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
        }
    }
}


constexpr size_t NUM_RAY_SAMPLE_POINTS = 256;


void sample_beckmann(float alpha) {

    UniformPathSampler sampler(0);
    sampler.next2D();

    Eigen::MatrixXd normals(10000000, 3);
    Microfacet::Distribution distribution("beckmann");

    for (int i = 0; i < normals.rows(); i++) {
        Vec3f normal = Microfacet::sample(distribution, alpha, sampler.next2D());
        normals(i, 0) = normal[0];
        normals(i, 1) = normal[1];
        normals(i, 2) = normal[2];
    }

    {
        std::ofstream xfile(incrementalFilename(Path(tinyformat::format("microfacet/visible-normals/beckmann-%.4f.bin", alpha)), "", false).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)normals.data(), sizeof(normals[0]) * normals.size());
        xfile.close();
    }

}

void v_ndf(std::shared_ptr<GaussianProcess> gp, float angle, int samples, std::string output) {

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(gp, 0, 1, 1, NUM_RAY_SAMPLE_POINTS);

    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    Ray ray = Ray(Vec3f(0.f, 0.f, 50.f), Vec3f(sin(angle), 0.f, -cos(angle)));

    ray.setNearT(-(ray.pos().z()-5.0f) / ray.dir().z());
    ray.setFarT(-(ray.pos().z()+5.0f) / ray.dir().z());

    Eigen::MatrixXd normals(samples, 3);

    int failed = 0;

    for (int s = 0; s < samples;) {

        if ((s + 1) % 100 == 0) {
            std::cout << s << "/" << samples << " - Failed: " << failed;
            std::cout << "\r";
        }

        Medium::MediumState state;
        state.reset();
        MediumSample sample;
        if (!gp_med->sampleDistance(sampler, ray, state, sample)) {
            failed++;
            continue;
        }

        sample.aniso.normalize();
        normals(s, 0) = sample.aniso.x();
        normals(s, 1) = sample.aniso.y();
        normals(s, 2) = sample.aniso.z();

        s++;
    }

    {
        std::ofstream xfile(incrementalFilename(Path(output) / Path(gp->_cov->id()) + Path(tinyformat::format("-%.1fdeg-%d.bin", 180 * angle / PI, NUM_RAY_SAMPLE_POINTS)), "", false).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)normals.data(), sizeof(normals[0]) * normals.size());
        xfile.close();
    }
}

void ndf(std::shared_ptr<GaussianProcess> gp, int samples, std::string output) {

    Path basePath = Path(output) / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(gp, 0, 1, 1, NUM_RAY_SAMPLE_POINTS);

    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    Eigen::MatrixXd normals(samples, 3);


    Ray ray = Ray(Vec3f(0.f, 0.f, 50.f), Vec3f(0.f, 0.f, -1.f));
    ray.setNearT(-5.f);
    ray.setFarT(5.f);

    for (int s = 0; s < samples;) {

        if ((s + 1) % 10000 == 0) {
            std::cout << s << "/" << samples;
            std::cout << "\r";
        }

        Medium::MediumState state;
        Vec3d grad;
        gp_med->sampleGradient(sampler, ray, Vec3d(-10.f, 20.0f, -5.f), state, grad);

        grad.normalize();
        normals(s, 0) = grad.x();
        normals(s, 1) = grad.y();
        normals(s, 2) = grad.z();

        s++;
    }

    {
        std::ofstream xfile(incrementalFilename(basePath + Path("-ndf.bin"), "", false).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)normals.data(), sizeof(normals[0]) * normals.size());
        xfile.close();
    }
}

void ndf_cond_validate(std::shared_ptr<GaussianProcess> gp, int samples, 
    std::string output, float angle = (2 * PI) / 8, 
    GPNormalSamplingMethod nsm = GPNormalSamplingMethod::ConditionedGaussian, float zrange = 4.f, 
    float maxStepSize = 0.15f, std::function<void(float)> update_progress = nullptr, int normalResampleCount = 1) {
    
    Path basePath = Path(output) / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    if (nsm == GPNormalSamplingMethod::Beckmann) {
        maxStepSize = 100000.f;
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(
        gp, 0, 1, 1, NUM_RAY_SAMPLE_POINTS, 
        nsm == GPNormalSamplingMethod::Beckmann ? GPCorrelationContext::Goldfish : GPCorrelationContext::Goldfish,
        nsm == GPNormalSamplingMethod::Beckmann ? GPIntersectMethod::Mean : GPIntersectMethod::GPDiscrete, 
        nsm);

    gp_med->loadResources();
    gp_med->prepareForRender();

    UniformPathSampler sampler(0);
    sampler.next2D();

    std::vector<Vec3d> sampledNormals(samples);

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle), 0.f, -cos(angle)));

    Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
    ray.setDir(mat.transformVector(ray.dir()));

    ray.setNearT(-(ray.pos().z() - zrange * 0.5f) / ray.dir().z());
    ray.setFarT(-(ray.pos().z() + zrange * 0.5f) / ray.dir().z());

    for (int s = 0; s < samples;) {

        if (update_progress && (s + 1) % 1000 == 0) {
            if (update_progress) {
                update_progress(float(s) / samples);
            }
        }

        Medium::MediumState state;
        state.reset();

        MediumSample sample;
        Ray tRay = ray;
        tRay.setFarT(tRay.nearT() + maxStepSize * NUM_RAY_SAMPLE_POINTS);
        bool success = gp_med->sampleDistance(sampler, tRay, state, sample);
        while (success && sample.exited) {
            tRay.setNearT(tRay.farT());
            tRay.setFarT(tRay.nearT() + maxStepSize * NUM_RAY_SAMPLE_POINTS);
            success = gp_med->sampleDistance(sampler, tRay, state, sample);
        }

        if (!success) {
            continue;
        }

        for (int i = 0; i < normalResampleCount; i++) {
            Vec3d grad;
            if(gp_med->sampleGradient(sampler, tRay, vec_conv<Vec3d>(sample.p), state, grad)) {
                grad.normalize();
                sampledNormals[s] = grad;
                s++;
            }
        }

        /*Vec3d grad = sample.aniso.normalized();
        normals(s, 0) = grad.x();
        normals(s, 1) = grad.y();
        normals(s, 2) = grad.z();
        s++;*/
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%.1fdeg-%d-%s-%.3f.bin", 180 * angle / PI, NUM_RAY_SAMPLE_POINTS, GaussianProcessMedium::normalSamplingMethodToString(nsm), zrange)), 
                "", false).asString(), 
            std::ios::out | std::ios::binary);

        xfile.write((char*)sampledNormals.data(), sizeof(sampledNormals[0]) * sampledNormals.size());
        xfile.close();
    }
}

template<typename T>
static std::shared_ptr<T> instantiate(JsonPtr value, const Scene& scene)
{
    auto result = StringableEnum<std::function<std::shared_ptr<T>()>>(value.getRequiredMember("type")).toEnum()();
    result->fromJson(value, scene);
    return result;
}

int main(int argc, char** argv) {
    auto lmean = std::make_shared<LinearMean>(Vec3d(0.f), Vec3d(0.f, 0.f, 1.f), 1.0f);

    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(argc > 1 ? argv[1] : "testing/microfacet/covariances-resample.json");
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return -1;
    }

    float aniso = 1.0;
    float angle =  0.;
    Scene scene;

    auto covs = (*document)["covariances"];
    if (!covs || !covs.isArray()) {
        std::cerr << "There should be a `covariances` array in the file\n";
        return -1;
    }

    for (int i = 0; i < covs.size(); i++) {
        const auto& jcov = covs[i];
        try {
            auto cov = instantiate<CovarianceFunction>(jcov, scene);
            cov->loadResources();
            //cov->_aniso[2] = aniso;

            std::cout << "============================================\n";
            std::cout << cov->id() << "\n";

            auto gp = std::make_shared<GaussianProcess>(lmean, cov);
            gp->_covEps = 0; // 0.00001f;
            gp->_maxEigenvaluesN = 1024;

            float alpha = gp->_cov->compute_beckmann_roughness();
            float bound = gp->noIntersectBound(Vec3d(0.), 0.9999);
            float maxStepsize = gp->goodStepsize(Vec3d(0.), 0.99);
            std::cout << "Beckmann roughness: " << alpha << "\n";
            std::cout << "0.9999 percentile: " << bound << "\n";
            std::cout << "Good stepsize: " << maxStepsize << "\n";

            std::cout << "Step sizes:\t";

            constexpr size_t NUM_STEPS = 3;

            for (int j = 0; j < NUM_STEPS; j++) {
                float angle = 80 + float(j) / (NUM_STEPS-1) * 9;
                angle = angle / 180 * PI;
                
                
                float travel_distance = 2 * bound / cos(angle);
                std::cout << travel_distance / NUM_RAY_SAMPLE_POINTS << ", ";
            }
            std::cout << "\n";

            std::cout << "cov(step):\t";

            for (int j = 0; j < NUM_STEPS; j++) {
                float angle = 80 + float(j) / (NUM_STEPS - 1) * 9;
                angle = angle / 180 * PI;

                float travel_distance = 2 * bound / cos(angle);
                std::cout << cov->operator()(Derivative::None, Derivative::None, Vec3d(0.), Vec3d(travel_distance / NUM_RAY_SAMPLE_POINTS, 0., 0.), Vec3d(0.), Vec3d(0.)) << ", ";
            }
            std::cout << "\n";

            //sample_beckmann(alpha);

            //normals_and_stuff(*gp, "microfacet/normals/");

            /*
            auto testFile = Path("microfacet/visible-normals/") / Path(gp->_cov->id()) + Path(tinyformat::format("-%.1fdeg-%d.bin", 180 * angle / PI, NUM_RAY_SAMPLE_POINTS));
            if (testFile.exists()) {
                std::cout << "skipping...\n";
                continue;
            }
            */

            float angleInDeg = 80.967;

            ndf_cond_validate(gp, 10000, "testing/microfacet/normals-resample/", 45.f / 180 * PI, GPNormalSamplingMethod::ConditionedGaussian, 1.0f, 0.15f, nullptr, 2500);

            //ndf_cond_validate(gp, 1000000, "testing/microfacet/normals-validate-nocond", angleInDeg / 180 * PI, GPNormalSamplingMethod::ConditionedGaussian, 0.02f);
            //ndf_cond_validate(gp, 10000000, "testing/microfacet/normals-validate-nocond", angleInDeg / 180 * PI, GPNormalSamplingMethod::Beckmann, 0.02f);
            //ndf_cond_validate(gp, 100000, "testing/microfacet/normals-validate-nocond", 3 * PI / 8, GPNormalSamplingMethod::ConditionedGaussian, 0.02f);
            //ndf_cond_validate(gp, 100000, "testing/microfacet/normals-validate-nocond", 3 * PI / 8, GPNormalSamplingMethod::ConditionedGaussian, 0.05f);
            //ndf_cond_validate(gp, 100000, "testing/microfacet/normals-validate-nocond", 3 * PI / 8, GPNormalSamplingMethod::ConditionedGaussian, 0.10f);
            //ndf_cond_validate(gp, 100000, "testing/microfacet/normals-validate-nocond", 3 * PI / 8, GPNormalSamplingMethod::ConditionedGaussian, 1.00f);
            //ndf_cond_validate(gp, 10000000, "testing/microfacet/normals-validate-nocond", 3 * PI / 8, GPNormalSamplingMethod::Beckmann, 0.2f);
            //ndf_cond_validate(gp, 1000000, "testing/microfacet/normals-validate-nocond", 3 * PI / 8, GPNormalSamplingMethod::Beckmann, 0.1f);

            /*
            MultiProgress<ProgressBar, NUM_STEPS> progress;
#pragma omp parallel for
            for (int j = 0; j < NUM_STEPS; j++) {
                float angle = 80 + float(j) / (NUM_STEPS - 1) * 9;
                angle = angle / 180 * PI;

                auto update_progress = [&progress, j](float p) {
                    progress.update(j, p);
                };

                ndf_cond_validate(gp, 100000, "testing/microfacet/smith-test/angle-range", angle, GPNormalSamplingMethod::ConditionedGaussian, bound * 2, maxStepsize, update_progress);
                ndf_cond_validate(gp, 1000000, "testing/microfacet/smith-test/angle-range", angle, GPNormalSamplingMethod::Beckmann, bound * 2, maxStepsize, update_progress);
            }*/
            
            //ndf(gp, 10000000, "microfacet/normals/");
            //side_view(gp, "microfacet/side-view/");
        }
        catch (std::exception& e) {
            std::cerr << e.what() << "\n";
        }
    }
}


