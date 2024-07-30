#include <core/math/GaussianProcess.hpp>
#include <core/media/FunctionSpaceGaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

using FloatDD = autodiff::dual2nd;
using Vec3DD = autodiff::Vector3dual2nd;
using VecXDD = autodiff::VectorXdual2nd;

FloatDD test_f(Vec3DD x, Vec3DD y) {
    return x.dot(y);
}

void directional_deriv_testing() {
    
    Vec3DD x = -Vec3DD::Ones();
    Vec3DD y = Vec3DD::Ones();

    auto hess = autodiff::hessian(test_f, wrt(x, y), at(x, y)).block(3, 0, 3, 3);
    auto grad = autodiff::gradient(test_f, wrt(x, y), at(x, y));
    std::cout << hess << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << grad.transpose() << std::endl;


    //Eigen::VectorXd dirX = 


}

void mvn_testing() {
    CovMatrix cov(2,2);
    cov.coeffRef(0, 0) = 1;
    cov.coeffRef(0, 1) = 0.5;
    cov.coeffRef(1, 0) = 0.5;
    cov.coeffRef(1, 1) = 1;

    Eigen::VectorXd mean(2);
    mean(0) = 1;
    mean(1) = 2;

    MultivariateNormalDistribution mvn(mean, cov.triangularView<Eigen::Lower>());

    UniformPathSampler sampler(0);
    sampler.next2D();
    auto samples = mvn.sample(nullptr, 0, 10000, sampler);

    std::vector<double> pdf;
    std::vector< Eigen::Vector2d> pdf_ps;

    for (int j = 0; j < 100; j++) {
        for (int i = 0; i < 100; i++) {
            Eigen::Vector2d p = { i / 99. * 9 - 3, j / 99. * 9 - 3 };
            pdf.push_back(mvn.eval(p));
            pdf_ps.push_back(p);
        }
    }

    Path basePath = Path("testing/basic/mvn");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    {
        std::ofstream xfile("testing/basic/mvn/samples.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(double) * samples.size());
        xfile.close();
    }

    {
        std::ofstream xfile("testing/basic/mvn/pdf.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)pdf.data(), sizeof(double) * pdf.size());
        xfile.close();
    }

    {
        std::ofstream xfile("testing/basic/mvn/pdf_points.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)pdf_ps.data(), sizeof(pdf_ps[0]) * pdf_ps.size());
        xfile.close();
    }
}

int main() {
    mvn_testing();
    return 0;

	auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.0f, 1.0f));

    FunctionSpaceGaussianProcessMedium gp_med(gp, 0, 1, 1, NUM_SAMPLE_POINTS);
    gp_med.prepareForRender();
    
    UniformPathSampler sampler(0);
    sampler.next2D();

    Ray ray(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.0f, 15.0f);
    
    std::vector<float> ts;

    float L0 = (*gp->_cov)(Derivative::None, Derivative::None, Vec3d(0.f), Vec3d(0.f), Vec3d(1.f, 0.f, 0.f), Vec3d(1.f, 0.f, 0.f));
    float L2 = (*gp->_cov)(Derivative::First, Derivative::First, Vec3d(0.f), Vec3d(0.f), Vec3d(1.f, 0.f, 0.f), Vec3d(1.f, 0.f, 0.f));

    std::cout << "L0:" << L0 << std::endl;
    std::cout << "L2:" << L2 << std::endl;

    int sample_count = 10000;
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
            std::ofstream xfile(tinyformat::format("transmittance/%s-sample-fp.bin", gp->_cov->id()), std::ios::out | std::ios::binary);
            xfile.write((char*)ts.data(), sizeof(float) * ts.size());
            xfile.close();
        }
    }


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
    }

}
