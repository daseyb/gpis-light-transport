#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 512;

int main() {

	GaussianProcess gp(std::make_shared<SphericalMean>(Vec3d(5.f, 0.5f, 0.f), 3.f), std::make_shared<SquaredExponentialCovariance>(1.0f, 1.0f));
    
    UniformPathSampler sampler(0);

    Ray ray(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.0f, 10.0f);
    auto rd = vec_conv<Vec3d>(ray.dir());

    std::array<Vec3d, NUM_SAMPLE_POINTS + 1> points;
    std::array<Derivative, NUM_SAMPLE_POINTS + 1> derivs;

    std::vector<float> ts;

    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        float t = lerp(ray.nearT(), ray.farT(), clamp((i - sampler.next1D()) / NUM_SAMPLE_POINTS, 0.f, 1.f));
        ts.push_back(t);
        points[i] = vec_conv<Vec3d>(ray.pos() + t * ray.dir());
        derivs[i] = Derivative::None;
    }

    points[NUM_SAMPLE_POINTS] = points[0];
    derivs[NUM_SAMPLE_POINTS] = Derivative::First;


    {
        std::vector<float> normalSamples;
        // Box muller transform
        for (int i = 0; i < 10000; i++) {
            Vec2d samples = rand_normal_2(sampler);
            normalSamples.push_back(samples.x());
            normalSamples.push_back(samples.y());
        }

        std::ofstream xfile("normalSamples.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)normalSamples.data(), sizeof(float) * normalSamples.size());
        xfile.close();
    }
    

    {
        std::ofstream xfile("ts.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)ts.data(), sizeof(float) * ts.size());
        xfile.close();
    }

    {
        Eigen::VectorXf mean = gp.mean(points.data(), derivs.data(), nullptr, rd, points.size()).cast<float>();
        std::ofstream xfile("mean.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        Eigen::Matrix4Xf kernel(4, NUM_SAMPLE_POINTS);

        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            kernel(0, i) = (*gp._cov)(Derivative::None, Derivative::None, points[0], points[i], rd, rd);
            kernel(1, i) = (*gp._cov)(Derivative::None, Derivative::First, points[0], points[i], rd, rd);
            kernel(2, i) = (*gp._cov)(Derivative::First, Derivative::None, points[0], points[i], rd, rd);
            kernel(3, i) = (*gp._cov)(Derivative::First, Derivative::First, points[0], points[i], rd, rd);
        }

        std::ofstream xfile("kernel-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)kernel.data(), sizeof(float) * kernel.rows() * kernel.cols());
        xfile.close();
    }

    {
        Eigen::MatrixXf cov = gp.cov(points.data(), points.data(), derivs.data(), derivs.data(), nullptr, nullptr, rd, points.size(), points.size()).cast<float>();
        std::ofstream xfile("cov.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)cov.data(), sizeof(float) * cov.rows() * cov.cols());
        xfile.close();
    }

    {
        Eigen::MatrixXf samples = gp.sample(points.data(), derivs.data(), points.size(), nullptr, nullptr, 0, rd, 50, sampler).cast<float>();
        std::ofstream xfile("samples.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Constraint, 1> constraints = { { 0, 0, 0, FLT_MAX } };
        Eigen::MatrixXf samples = gp.sample(points.data(), derivs.data(), points.size(), nullptr, constraints.data(), constraints.size(), rd, 50, sampler).cast<float>();
        std::ofstream xfile("samples-free-space.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Constraint, 1> constraints = { { 0, 0, 0, FLT_MAX } };

        Eigen::MatrixXd samples = gp.sample(
            points.data(), derivs.data(), points.size(), nullptr,
            constraints.data(), constraints.size(),
            rd, 50000, sampler);

        std::vector<float> sampleTs;
        for (int s = 0; s < samples.cols(); s++) {
            float prevV = (float)samples(0, s);
            float prevT = ts[0];
            for (int p = 1; p < NUM_SAMPLE_POINTS; p++) {
                float currV = (float)samples(p, s);
                float currT = ts[p];
                if (currV < 0) {
                    float offsetT = prevV / (prevV - currV);
                    float t = lerp(prevT, currT, offsetT);
                    sampleTs.push_back(t);
                    //sample.aniso = gpSamples(p * 2, 0);
                    break;
                }
                prevV = currV;
                prevT = currT;
            }
        }

        std::ofstream xfile("dist-samples-free.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)sampleTs.data(), sizeof(float) * sampleTs.size());
        xfile.close();
    }

    {
        std::array<Vec3d, 2> cond_pts = { points[0]};
        std::array<Derivative, 2> cond_deriv = { Derivative::None };
        std::array<double, 2> cond_vs = { 0 };

        Eigen::MatrixXf samples = gp.sample_cond(
            points.data(), derivs.data(), points.size(), nullptr,
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
            nullptr, 0,
            rd, 50, sampler).cast<float>();

        std::ofstream xfile("samples-cond.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Vec3d, 2> cond_pts = { points[0], points[0] };
        std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };
        std::array<double, 2> cond_vs = { 0, 1 };
        std::array<Constraint, 0> constraints = {  };


        Eigen::MatrixXf samples = gp.sample_cond(
            points.data(), derivs.data(), points.size(), nullptr,
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
            constraints.data(), constraints.size(),
            rd, 50, sampler).cast<float>();

        std::ofstream xfile("samples-cond-const.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Vec3d, 2> cond_pts = { points[0], points[0] };
        std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };
        std::array<double, 2> cond_vs = { 0, 1 };
        std::array<Constraint, 0> constraints = {  };

        Eigen::MatrixXf samples = gp.sample_cond(
            points.data(), derivs.data(), points.size(), nullptr,
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
            constraints.data(), constraints.size(),
            rd, 50000, sampler).cast<float>();

        std::vector<float> sampleTs;
        for (int s = 0; s < samples.cols(); s++) {
            float prevV = samples(0, s);
            float prevT = ts[0];
            for (int p = 1; p < NUM_SAMPLE_POINTS; p++) {
                float currV = samples(p, s);
                float currT = ts[p];
                if (currV < 0) {
                    float offsetT = prevV / (prevV - currV);
                    float t = lerp(prevT, currT, offsetT);
                    sampleTs.push_back(t);
                    //sample.aniso = gpSamples(p * 2, 0);
                    break;
                }
                prevV = currV;
                prevT = currT;
            }
        }

        std::ofstream xfile("dist-samples-cond.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)sampleTs.data(), sizeof(float) * sampleTs.size());
        xfile.close();
    }



    //tfile.close();


}
