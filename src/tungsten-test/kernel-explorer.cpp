#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/math/WeightSpaceGaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>
#include <thread/ThreadUtils.hpp>
#include <io/Scene.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

template<typename T>
static std::shared_ptr<T> instantiate(JsonPtr value, const Scene& scene)
{
    auto result = StringableEnum<std::function<std::shared_ptr<T>()>>(value.getRequiredMember("type")).toEnum()();
    result->fromJson(value, scene);
    return result;
}


constexpr size_t NUM_SAMPLE_POINTS = 32;

std::tuple<std::vector<Vec3d>, std::vector<Vec3d>, std::vector<double>, std::vector<Derivative>> sample_surface(float scale, bool doNormal) {

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ns;
    std::vector<double> vs;
    std::vector<Derivative> ds;

    Vec3d c = Vec3d(0.75, 0, 0.f);
    double r = 0.4f;
    
    if (!doNormal) {
        ps.push_back(c);
        ns.push_back(Vec3d(0.f));
        vs.push_back(-r);
        ds.push_back(Derivative::None);

        ps.push_back(c + r * 2 * Vec3d(-1.f, 1.f, 0.f).normalized());
        ns.push_back(Vec3d(0.f));
        vs.push_back(r);
        ds.push_back(Derivative::None);
    }

    int num_pts = 5;
    for (int i = 0; i < num_pts; i++) {
        float a = lerp(PI / 2, PI * 1.5f, float(i) / (num_pts - 1));
        Vec3d p = c + Vec3d(cos(a), sin(a), 0.f) * r;
        ps.push_back(p * scale);
        ns.push_back((p - c).normalized());
        vs.push_back(0);
        ds.push_back(Derivative::None);
        
        if (doNormal) {
            ps.push_back(p * scale);
            ns.push_back((p - c).normalized());
            vs.push_back(1);
            ds.push_back(Derivative::First);
        }
    }

    return { ps, ns, vs, ds };
}

void eval_kernel(std::shared_ptr<CovarianceFunction> cov) {
    std::vector<double> kernel;
    std::vector<double> derive1;
    std::vector<double> derive2;

    for (int i = 0; i < 500; i++) {
        double px = 2. * (double(i) / (500 - 1));
        kernel.push_back((*cov)(Derivative::None, Derivative::None, Vec3d(0.), Vec3d(px, 0., 0.), Vec3d(0.), Vec3d(0.)));
        derive1.push_back((*cov)(Derivative::None, Derivative::First, Vec3d(0.), Vec3d(px, 0., 0.), Vec3d(0.), Vec3d(1., 0., 0.)));
        derive2.push_back((*cov)(Derivative::First, Derivative::First, Vec3d(0.), Vec3d(px, 0., 0.), Vec3d(1., 0., 0.), Vec3d(1., 0., 0.)));
    }

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/kernel.bin", cov->id()), std::ios::out | std::ios::binary);
        xfile.write((char*)kernel.data(), sizeof(double) * kernel.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/kernel-dx.bin", cov->id()), std::ios::out | std::ios::binary);
        xfile.write((char*)derive1.data(), sizeof(double) * derive1.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/kernel-dxdy.bin", cov->id()), std::ios::out | std::ios::binary);
        xfile.write((char*)derive2.data(), sizeof(double) * derive2.size());
        xfile.close();
    }
}

void realization_ws(std::shared_ptr<CovarianceFunction> cov, int n) {

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), cov);

    std::cout << gp->compute_beckmann_roughness(Vec3d(0.)) << "\n";


    {
        UniformPathSampler sampler(0);
        sampler.next2D();

        auto wsgp = WeightSpaceBasis::sample(cov, n, sampler);

        auto real = wsgp.sampleRealization(gp, sampler);

        int res = 300;
        std::vector<Vec3d> points(res * res);
        std::vector<Derivative> derivs(res * res, Derivative::None);

        Eigen::VectorXd samples;
        samples.resize(res * res);

        {
            int idx = 0;
            for (int i = 0; i < res; i++) {
                double px = 2. * (double(i) / (res - 1) - 0.5);

                for (int j = 0; j < res; j++) {
                    double py = 2. * (double(j) / (res - 1) - 0.5);
                    points[idx] = Vec3d(px, py, 0.);

                    samples[idx] = real.evaluate(points[idx]);
                    idx++;
                }
            }


            {
                std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-grid-samples-ws.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }

        }
    }
}

void realization_fs(std::shared_ptr<CovarianceFunction> cov) {

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), cov);

    std::cout << gp->compute_beckmann_roughness(Vec3d(0.)) << "\n";


    {
        UniformPathSampler sampler(0);
        sampler.next2D();

        std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
        std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    points[idx][2] = 0.f;
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }


            auto [samples, gpIds] = gp->sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), 1, sampler)->flatten();

            {
                std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-grid-samples-fs.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
}


void conditioned_realization(std::shared_ptr<CovarianceFunction> cov, bool doNormals) {

    constexpr size_t NUM_SAMPLE_POINTS = 32;

    int num_reals = 2000;

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), cov);

    std::cout << gp->compute_beckmann_roughness(Vec3d(0.)) << "\n";

    auto [cps, cns, cvs, cds] = sample_surface(1.f, doNormals);

    gp->setConditioning(
        cps,
        cds,
        cns,
        cvs
    );

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-cond-ps-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondPs.data(), sizeof(Vec3d) * gp->_globalCondPs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-cond-ds-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondDerivs.data(), sizeof(Derivative) * gp->_globalCondDerivs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-cond-ns-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondDerivDirs.data(), sizeof(Vec3d) * gp->_globalCondDerivDirs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-cond-vs-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondValues.data(), sizeof(double) * gp->_globalCondValues.size());
        xfile.close();
    }

    {
        UniformPathSampler sampler(0);
        sampler.next2D();


        std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
        std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    points[idx][2] = 0.f;
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }


            auto [samples, gpIds] = gp->sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), num_reals, sampler)->flatten();

            {
                std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/%d-grid-samples-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
}



int main(int argc, const char** argv) {

    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(Path(argv[1]));
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return -1;
    }

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

            std::cout << "============================================\n";
            std::cout << cov->id() << "\n";

            Path basePath = Path("testing/kernel-explorer") / cov->id();
            if (!basePath.exists()) {
                FileUtils::createDirectory(basePath);
            }

            {
                std::ofstream xfile(tinyformat::format("testing/kernel-explorer/%s/label.txt", cov->id()), std::ios::out);
                std::string label;
                jcov.getField("name", label);
                xfile << label;
                xfile.close();
            }


            eval_kernel(cov);
            realization_fs(cov);
            if (cov->hasAnalyticSpectralDensity()) {
                realization_ws(cov, 3000);
            }
            //conditioned_realization(cov, cov->id().find("tp") == std::string::npos);
            conditioned_realization(cov, true);
        }
        catch (std::exception& e) {
            std::cerr << e.what() << "\n";
        }
    }
}
