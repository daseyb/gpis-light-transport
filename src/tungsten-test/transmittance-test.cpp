#include <core/math/GaussianProcess.hpp>
#include <core/media/FunctionSpaceGaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

float rices_formula(float u, float L0, float L2) {
    return exp(-u * u / (2 * L0)) * sqrt(L2 / L0) / (2 * PI);
}

void sample_ffds(std::shared_ptr<GaussianProcess> gp) {
    float meanv = gp->_mean->operator()(Derivative::None, Vec3d(0.f), Vec3d(1.f));
    FunctionSpaceGaussianProcessMedium gp_med(gp, 0, 1, 1, NUM_SAMPLE_POINTS);
    gp_med.prepareForRender();

    UniformPathSampler sampler(0);
    sampler.next2D();

    Ray ray(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.0f, 16.0f);

    std::vector<float> ts;

    float L0 = (*gp->_cov)(Derivative::None, Derivative::None, Vec3d(0.f), Vec3d(0.f), Vec3d(1.f, 0.f, 0.f), Vec3d(1.f, 0.f, 0.f));
    float L2 = (*gp->_cov)(Derivative::First, Derivative::First, Vec3d(0.f), Vec3d(0.f), Vec3d(1.f, 0.f, 0.f), Vec3d(1.f, 0.f, 0.f));

    std::cout << "L0:" << L0 << std::endl;
    std::cout << "L2:" << L2 << std::endl;
    std::cout << "R:" << rices_formula(meanv, L0, L2) << std::endl;


    Path basePath = Path("testing/transmittance") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    std::string filename = basePath.asString() + tinyformat::format("/sample-fp-%f.bin", meanv);

    if (Path(filename).exists()) {
        //continue;
    }

    int sample_count = 1000000;
    std::cout << "fp" << std::endl;
    {
        std::vector<float> ts;
        for (int i = 0; i < sample_count; i++) {
            if (i % 100 == 0) {
                std::cout << i << "/" << sample_count << "\r";
            }

            Medium::MediumState state;
            state.reset();
            MediumSample sample;
            if (gp_med.sampleDistance(sampler, ray, state, sample)) {
                ts.push_back(sample.continuedT);
            }
        }

        {
            std::ofstream xfile(
                filename,
                std::ios::out | std::ios::binary);
            xfile.write((char*)ts.data(), sizeof(float) * ts.size());
            xfile.close();
        }
    }
}

int main() {
    
    float meanv = 0;
    float length_scale = 1.0f;

    /*for (auto meanv : {-1.6f, 0.f, 1.6f}) {
        for (auto length_scale : { 2.0f, 0.25f, 0.125f }) {
            std::cout << "=========" << meanv << "-" << length_scale << "=======\n";
            auto mean = std::make_shared<HomogeneousMean>(meanv);
            auto gp = std::make_shared<GaussianProcess>(mean, std::make_shared<SquaredExponentialCovariance>(1.0f, length_scale));
        }
    }*/

    for (auto a : { 1.0f, 0.75f, 0.5f, 0.25f, 0.1f }) {
        std::cout << "=========" << a << "-" << length_scale << "=======\n";
        auto mean = std::make_shared<HomogeneousMean>(meanv);
        auto gp = std::make_shared<GaussianProcess>(mean, std::make_shared<RationalQuadraticCovariance>(1.0f, length_scale, a));
        sample_ffds(gp);
    }


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
