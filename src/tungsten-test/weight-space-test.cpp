#include <core/media/FunctionSpaceGaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>
#include <sampling/SampleWarp.hpp>
#include <core/math/WeightSpaceGaussianProcess.hpp>

#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>
#include "math/TangentFrame.hpp"

#include "io/CliParser.hpp"

#include <core/media/GaussianProcessMedium.hpp>
#include <cfloat>
#include <io/ImageIO.hpp>
#include <io/FileUtils.hpp>
#include <bsdfs/Microfacet.hpp>
#include <rapidjson/document.h>
#include <io/JsonDocument.hpp>
#include <io/JsonObject.hpp>
#include <io/Scene.hpp>
#include <math/GaussianProcessFactory.hpp>
#include <thread/ThreadUtils.hpp>
#include <phasefunctions/BRDFPhaseFunction.hpp>

using namespace Tungsten;

void compute_realization(std::shared_ptr<GaussianProcess> gp, size_t res, const WeightSpaceRealization& real) {

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }


    std::vector<Vec3d> points(res * res);
    std::vector<Derivative> derivs(res * res, Derivative::None);

    Eigen::VectorXd samples;
    samples.resize(res * res);

    std::vector<double> grid_xs(res);

    {
        int idx = 0;
        for (int i = 0; i < res; i++) {
            double px = 20. * (double(i) / (res - 1) - 0.5);
            grid_xs[i] = px;

            for (int j = 0; j < res; j++) {
                double py = 20. * (double(j) / (res - 1) - 0.5);
                points[idx] = Vec3d(px, py, 0.);

                samples[idx] = real.evaluate(points[idx]);
                idx++;
            }
        }


        {
            std::ofstream xfile(
                (basePath / Path("grid-samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
            xfile.close();
        }

        {
            std::ofstream xfile(
                (basePath / Path("grid-coordinates.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)grid_xs.data(), sizeof(double) * grid_xs.size());
            xfile.close();
        }
    }

}

double largestSphereAA(std::function<Affine<1>(Affine<3>)> implicit, Vec3d center) {
    const double delta = 0.001;

    double lower = 0.;
    double upper = 100.;

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;

    for (int i = 0; i < 1000; i++) {
        if (upper - lower < delta) {
            return lower;
        }

        double midpoint = (upper + lower) * 0.5;
        auto vs = {
            Vec3d(midpoint, 0., 0.),
            Vec3d(0., midpoint, 0.)
        };

        auto val = implicit(Affine<3>(center, vs));

        if (val.rangeBound() != RangeBound::Unknown) {
            lower = midpoint;
        }
        else {
            upper = midpoint;
        }
    }
    std::cerr << "Ran out of iterations in largest sphere AA." << std::endl;
    return lower;
}

double largestSphereAA(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, Vec3d center) {
    const double delta = 0.001;

    double lower = 0.;
    double upper = 100.;

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;

    for (int i = 0; i < 1000; i++) {
        if (upper - lower < delta) {
            return lower;
        }

        double midpoint = (upper + lower) * 0.5;
        auto vs = {
            Vec3d(midpoint, 0., 0.),
            Vec3d(0., midpoint, 0.)
        };

        if (real.rangeBound(center, vs) != RangeBound::Unknown) {
            lower = midpoint;
        }
        else {
            upper = midpoint;
        }
    }
    std::cerr << "Ran out of iterations in largest sphere AA." << std::endl;
    return lower;
}

void wos(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, Vec3d p, PathSampleGenerator& sampler) {
    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }
    std::vector<Vec3d> ps;
    std::vector<double> ds;

    double d = largestSphereAA(gp, real, p);
    int it = 0;

    ps.push_back(p);
    ds.push_back(d);


    while (++it < 10000 && abs(d) > 0.001) {
        auto xi = sampler.next2D();
        auto samp = SampleWarp::uniformCylinder(xi);
        samp.z() = 0;
        p += vec_conv<Vec3d>(samp) * d ;
        d = largestSphereAA(gp, real, p);

        ps.push_back(p);
        ds.push_back(d);
    }

    {
        std::ofstream xfile(
            (basePath / Path("wos-centers.bin")).asString(),
            std::ios::out | std::ios::binary);
        xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            (basePath / Path("wos-radii.bin")).asString(),
            std::ios::out | std::ios::binary);
        xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
        xfile.close();
    }

}


bool intersectRayAA(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, const Ray& ray, Vec3d& p) {
    const double sig_0 = (ray.farT() - ray.nearT()) * 0.1f;
    const double delta = 0.01;
    const double np = 1.5;
    const double nm = 0.5;

    double t = 0;
    double sig = sig_0;

    auto rd = vec_conv<Vec3d>(ray.dir());

    p = vec_conv<Vec3d>(ray.pos()) + t * rd;
    double f0 = real.evaluate(p);

    int sign0 = f0 < 0 ? -1 : 1;

    for (int i = 0; i < 2048 * 4; i++) {
        auto p_c = p + (t + ray.nearT() + delta) * rd;
        double f_c = real.evaluate(p_c);
        int signc = f_c < 0 ? -1 : 1;

        if (signc != sign0) {
            p = p_c;
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

        if (t >= ray.farT()) {
            return false;
        }
    }

    std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
    return false;
}

bool intersectRayAAWriteData(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, const Ray& ray, Vec3d& p) {
    const double sig_0 = (ray.farT() - ray.nearT()) * 0.1f;
    const double delta = 0.001;
    const double np = 1.5;
    const double nm = 0.5;
    
    double t = 0;
    double sig = sig_0;

    auto rd = vec_conv<Vec3d>(ray.dir());

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;
    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    p = vec_conv<Vec3d>(ray.pos()) + t * rd;
    double f0 = real.evaluate(p);

    int sign0 = f0 < 0 ? -1 : 1;

    for (int i = 0; i < 2048 * 4; i++) {
        auto p_c = p + (t + ray.nearT() + delta) * rd;
        double f_c = real.evaluate(p_c);
        int signc = f_c < 0 ? -1 : 1;


        if (signc != sign0) {
            ps.push_back(p_c);
            ds.push_back(rd * 0.000001);
            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-centers.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-extends.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return true;
        }

        auto c = p + (t + ray.nearT() + sig * 0.5) * rd;
        auto v = sig * 0.5 * rd;

        double nsig;
        if (real.rangeBound(c, { v }) != RangeBound::Unknown) {
            nsig = sig;
            sig = sig * np;

            ps.push_back(c);
            ds.push_back(v);
        }
        else {
            nsig = 0;
            sig = sig * nm;
        }

        t += max(nsig*0.98, delta);

        if (t >= ray.farT()) {
            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-centers.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-extends.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return false;
        }
    }

    std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
    return false;
}

bool intersectRaySphereTrace(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, const Ray& ray, Vec3d& p) {
    double t = ray.nearT() + 0.0001;
    double L = real.lipschitz();

    std::vector<Vec3d> ps;
    std::vector<double> ds;

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    for (int i = 0; i < 2048 * 4; i++) {
        p = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
        double f = real.evaluate(p) / L;

        ps.push_back(p);
        ds.push_back(f);

        if (f < 0.000000001) {

            {
                std::ofstream xfile(
                    (basePath / Path("ray-points.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("ray-distances.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return true;
        }

        t += f;

        if (t >= ray.farT()) {
            return false;
        }
    }

    std::cerr << "Ran out of iterations in mean intersect sphere trace." << std::endl;
    return false;
}

void compute_spectral_density(std::shared_ptr<GaussianProcess> gp) {
    gp->loadResources();

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    {
        std::vector<double> spectralDensity;
        size_t num_samples = 1000;
        double max_w = 10;
        for (size_t i = 0; i < num_samples; i++) {
            spectralDensity.push_back(gp->_cov->spectral_density(double(i) / num_samples * max_w));
        }

        {
            std::ofstream xfile(
                (basePath / Path("spectral_density.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)spectralDensity.data(), sizeof(double) * spectralDensity.size());
            xfile.close();
        }
    }


    {
        UniformPathSampler sampler(0);
        sampler.next2D();

        std::vector<double> spectralDensitySamples;
        size_t num_samples = 100000;
        for (size_t i = 0; i < num_samples; i++) {
            spectralDensitySamples.push_back(gp->_cov->sample_spectral_density(sampler));
        }

        {
            std::ofstream xfile(
                (basePath / Path("spectral_density_samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)spectralDensitySamples.data(), sizeof(double) * spectralDensitySamples.size());
            xfile.close();
        }
    }
}


void gen_data(std::shared_ptr<GaussianProcess> gp) {
    std::cout << gp->_cov->id() << "\n";
    
    compute_spectral_density(gp);
    
    UniformPathSampler sampler(0);
    sampler.next2D();

    auto basis = WeightSpaceBasis::sample(gp->_cov, 300, sampler);
    auto real = basis.sampleRealization(gp, sampler);

    std::cout << "L = " << real.lipschitz() << "\n";
    compute_realization(gp, 512, real);

    Ray r(Vec3f(-9.f, -9.f, 0.f), Vec3f(1.f, 1.f, 0.f).normalized(), 0.f, 100.f);
    Vec3d ip;
    intersectRaySphereTrace(gp, real, r, ip);
    intersectRayAAWriteData(gp, real, r, ip);
    wos(gp, real, Vec3d(1., 0., 0.), sampler);
}

void test_affine() {

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(100.f, 1.f, 0.1f));

    Affine<3> p(Vec3d(0., 0., 0.), { Vec3d(10., 0., 0.), Vec3d(0., 10., 0.) });

    UniformPathSampler sampler(0);
    sampler.next2D();
    WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, 2, sampler);

    auto real = basis.sampleRealization(gp, sampler);


    Path basePath = Path("testing/weight-space-affine") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    auto implicit = [&real](Affine<3> p) {
        return real.evaluate(p);
    };

    auto result = implicit(p);

    size_t res = 100;
    std::vector<Vec3d> points(res * res);

    Eigen::VectorXd samples;
    samples.resize(res * res);
    std::vector<double> grid_xs(res);

    auto pbs = p.mayContainBounds();
    {
        int idx = 0;
        for (int i = 0; i < res; i++) {
            for (int j = 0; j < res; j++) {
                points[idx] = vec_conv<Vec3d>(
                    lerp((Eigen::Array3d)pbs.lower,
                        (Eigen::Array3d)pbs.upper,
                        Eigen::Array3d(double(i) / (res - 1), double(j) / (res - 1), 0.)));

                samples[idx] = real.evaluate(points[idx]);
                idx++;
            }
        }

        {
            std::ofstream xfile(
                (basePath / Path("samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
            xfile.close();
        }
    }

    {
        std::ofstream xfile(
            (basePath / Path("result.bin")).asString(),
            std::ios::out | std::ios::binary);
        xfile.write((char*)result.base.data(), sizeof(double) * result.base.size());
        xfile.write((char*)result.aff.data(), sizeof(double) * result.aff.size());
        xfile.write((char*)result.err.data(), sizeof(double) * result.err.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            (basePath / Path("bounds.bin")).asString(),
            std::ios::out | std::ios::binary);
        auto bounds = result.mayContainBounds();
        xfile.write((char*)bounds.lower.data(), sizeof(double) * bounds.lower.size());
        xfile.write((char*)bounds.upper.data(), sizeof(double) * bounds.upper.size());
        xfile.close();
    }

    std::cout << "===========================\n";
    std::cout << largestSphereAA(implicit, Vec3d(0., 0., 0.)) << "\n";

}

Vec3d sample_beckmann_vndf(Ray ray, Vec3d normal, std::shared_ptr<GaussianProcess> _gp) {
    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

    Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
    float alpha = _gp->_cov->compute_beckmann_roughness();
    BeckmannNDF ndf(0, alpha, alpha);

    return vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
}


void microfacet_sample_beckmann(std::shared_ptr<GaussianProcess> gp, Path outputDir, int samples, float angle, double zrange, size_t seed, std::string tag) {

    Path basePath = outputDir / Path("beckmann") / Path(gp->_cov->id());

    g_mt = decltype(g_mt)(seed);

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle / 180 * PI), 0.f, -cos(angle / 180 * PI)));
    {
        Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
        ray.setDir(mat.transformVector(ray.dir()).normalized());
        ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
        ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());
    }

    std::string filename_normals = basePath.asString() + tinyformat::format("/%.1fdeg-%snormals-%d.bin", angle, tag, seed);

    std::vector<Vec3d> sampledNormals(samples);

    ThreadUtils::parallelFor(0, samples, 32, [&](auto s) {
        UniformPathSampler sampler(MathUtil::hash32(seed) + s);
        sampler.next2D();
        sampledNormals[s] = sample_beckmann_vndf(ray, Vec3d(0., 0., 1.), gp);
    });

    {
        std::ofstream xfile(
            filename_normals,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledNormals.data(), sizeof(sampledNormals[0]) * sampledNormals.size());
        xfile.close();
    }
}

void microfacet_intersect_test(std::shared_ptr<GaussianProcess> gp, Path outputDir, int samples, int numBasisFs, float angle, double zrange, size_t seed, std::string tag) {

    Path basePath = outputDir / Path("weight-space") / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle / 180 * PI), 0.f, -cos(angle / 180 * PI)));

    Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
    ray.setDir(mat.transformVector(ray.dir()).normalized());

    ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
    ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());

    std::string filename_normals = basePath.asString() + tinyformat::format("/%.1fdeg-%d-%snormals-%d.bin", angle, numBasisFs, tag, seed);

    std::vector<Vec3d> sampledNormals(samples);

    ThreadUtils::parallelFor(0, samples, 32, [&](auto s) {
        UniformPathSampler sampler(MathUtil::hash32(seed) + s);
        sampler.next2D();

        auto basis = WeightSpaceBasis::sample(gp->_cov, numBasisFs, sampler);
        auto real = basis.sampleRealization(gp, sampler);

        Vec3d ip;
        if (intersectRayAA(gp, real, ray, ip)) {
            Vec3d grad = real.evaluateGradient(ip);
            sampledNormals[s]= grad.normalized();
        }
        else
        {
            std::cout << "Error!\n";
        }
    });


    {
        std::ofstream xfile(
            filename_normals,
            std::ios::out | std::ios::binary);
        xfile.write((char*)sampledNormals.data(), sizeof(sampledNormals[0]) * sampledNormals.size());
        xfile.close();
    }
}


void ndf_cond_validate(std::shared_ptr<GaussianProcess> gp, int samples, int seed, Path outputDir, float angle = (2 * PI) / 8, GPNormalSamplingMethod nsm = GPNormalSamplingMethod::ConditionedGaussian, float zrange = 4.f, float maxStepSize = 0.15f, int numRaySamplePoints = 64, std::string tag = "") {
    
    Path basePath = outputDir / Path("function-space") / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    if (nsm == GPNormalSamplingMethod::Beckmann) {
        maxStepSize = 100000.f;
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(
        gp, 0, 1, 1, numRaySamplePoints, 
        nsm == GPNormalSamplingMethod::Beckmann ? GPCorrelationContext::Goldfish : GPCorrelationContext::Goldfish,
        nsm == GPNormalSamplingMethod::Beckmann ? GPIntersectMethod::Mean : GPIntersectMethod::GPDiscrete, 
        nsm);

    gp_med->loadResources();
    gp_med->prepareForRender();

    UniformPathSampler sampler(seed);
    sampler.next2D();

    std::vector<Vec3d> sampledNormals(samples);

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle / 180 * PI), 0.f, -cos(angle / 180 * PI)));

    {
        Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
        ray.setDir(mat.transformVector(ray.dir()));
        ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
        ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());
    }

    if(maxStepSize == 0) {
        maxStepSize = (ray.farT() - ray.nearT()) / numRaySamplePoints;
    }

    for (int s = 0; s < samples;) {
        Medium::MediumState state;
        state.reset();

        MediumSample sample;
        Ray tRay = ray;
        tRay.setFarT(tRay.nearT() + maxStepSize * numRaySamplePoints);
        bool success = gp_med->sampleDistance(sampler, tRay, state, sample);
        while (success && sample.exited) {
            tRay.setNearT(tRay.farT());
            tRay.setFarT(tRay.nearT() + maxStepSize * numRaySamplePoints);
            success = gp_med->sampleDistance(sampler, tRay, state, sample);
        }

        if (!success) {
            continue;
        }

        sampledNormals[s] = sample.aniso.normalized();
        s++;
    }

    {
        std::ofstream xfile(
            (basePath + Path(tinyformat::format("/%.1fdeg-%d-%s-%.3f-%.3f-%snormals-%d.bin", angle, numRaySamplePoints, GaussianProcessMedium::normalSamplingMethodToString(nsm), zrange, maxStepSize, tag, seed))).asString(), 
            std::ios::out | std::ios::binary);

        xfile.write((char*)sampledNormals.data(), sizeof(sampledNormals[0]) * sampledNormals.size());
        xfile.close();
    }
}


void bsdf_sample(std::shared_ptr<GaussianProcess> gp, std::shared_ptr<BRDFPhaseFunction> phase, int samples, int seed, Path outputDir, float angle = (2 * PI) / 8, GPNormalSamplingMethod nsm = GPNormalSamplingMethod::ConditionedGaussian, float zrange = 4.f, float maxStepSize = 0.15f, int numRaySamplePoints = 64, std::string tag = "") {

    Path basePath = outputDir / Path("function-space") / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    if (nsm == GPNormalSamplingMethod::Beckmann) {
        maxStepSize = 100000.f;
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(
        gp, 0, 1, 1, numRaySamplePoints,
        nsm == GPNormalSamplingMethod::Beckmann ? GPCorrelationContext::Goldfish : GPCorrelationContext::Goldfish,
        nsm == GPNormalSamplingMethod::Beckmann ? GPIntersectMethod::Mean : GPIntersectMethod::GPDiscrete,
        nsm);

    gp_med->loadResources();
    gp_med->prepareForRender();

    UniformPathSampler sampler(seed);
    sampler.next2D();

    std::vector<Vec3d> sampledNormals(samples);

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle / 180 * PI), 0.f, -cos(angle / 180 * PI)));

    {
        Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
        ray.setDir(mat.transformVector(ray.dir()));
        ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
        ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());
    }

    if (maxStepSize == 0) {
        maxStepSize = (ray.farT() - ray.nearT()) / numRaySamplePoints;
    }

    for (int s = 0; s < samples;) {
        Medium::MediumState state;
        state.reset();

        MediumSample sample;
        Ray tRay = ray;
        tRay.setFarT(tRay.nearT() + maxStepSize * numRaySamplePoints);
        bool success = gp_med->sampleDistance(sampler, tRay, state, sample);
        while (success && sample.exited) {
            tRay.setNearT(tRay.farT());
            tRay.setFarT(tRay.nearT() + maxStepSize * numRaySamplePoints);
            success = gp_med->sampleDistance(sampler, tRay, state, sample);
        }

        if (!success) {
            continue;
        }

        PhaseSample ps;
        if (!phase->sample(sampler, ray.dir(), sample, ps)) {
            continue;
        }

        sampledNormals[s] = vec_conv<Vec3d>(ps.w);
        s++;
    }

    {
        std::ofstream xfile(
            (basePath + Path(tinyformat::format("/%.1fdeg-%d-%s-%.3f-%.3f-%sphase-%d.bin", angle, numRaySamplePoints, GaussianProcessMedium::normalSamplingMethodToString(nsm), zrange, maxStepSize, tag, seed))).asString(),
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

int main(int argc, const char** argv) {

    if(argc == 1) {
        //test_affine();
        //return 0;
        gen_data(
            std::make_shared<GaussianProcess>(
                std::make_shared<LinearMean>(Vec3d(0., 0., 0.), Vec3d(0., 0., 1.), 1.f), std::make_shared<MaternCovariance>(1.f, 0.25f, 2.5f)));

        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(1.f, 1.f, 0.1f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(10.f, 1.f, 0.1f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(100.f, 1.f, 0.1f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(10.f, 1.f, 0.1f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(1.f, 0.2f, 0.1f)));


        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.f, 0.5f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 1.0f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 0.5f)));
        //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 5.0f)));
    } else {

        static const int OPT_THREADS           = 1;
        static const int OPT_ANGLE             = 2;
        static const int OPT_HELP              = 3;
        static const int OPT_BASIS             = 4;
        static const int OPT_OUTPUT_DIRECTORY  = 5;
        static const int OPT_SPP               = 6;
        static const int OPT_SEED              = 7;
        static const int OPT_BECKMANN          = 8;
        static const int OPT_WEIGHTSPACE       = 9;
        static const int OPT_FUNCTIONSPACE     = 10;
        static const int OPT_INPUT_DIRECTORY   = 11;
        static const int OPT_RAYSAMPLES        = 12;
        static const int OPT_MAX_STEPSIZE      = 13;
        static const int OPT_TAG               = 14;
        static const int OPT_BOUND             = 15;
        static const int OPT_SOLVER_THRESHOLD  = 16;

        CliParser parser("tungsten", "[options] covariances1 [covariances2 [covariances3...]]");

        parser.addOption('h', "help", "Prints this help text", false, OPT_HELP);
        parser.addOption('t', "threads", "Specifies number of threads to use (default: number of cores minus one)", true, OPT_THREADS);
        parser.addOption('i', "input-directory", "Specifies the input directory", true, OPT_INPUT_DIRECTORY);
        parser.addOption('d', "output-directory", "Specifies the output directory. Overrides the setting in the scene file", true, OPT_OUTPUT_DIRECTORY);
        parser.addOption('\0', "spp", "Sets the number of samples per pixel to render at. Overrides the setting in the scene file", true, OPT_SPP);
        parser.addOption('s', "seed", "Specifies the random seed to use", true, OPT_SEED);
        parser.addOption('a', "angle", "Ray angle in degrees (0 orthogonal, 90 parallel)", true, OPT_ANGLE);
        parser.addOption('b', "basis", "Number of basis functions", true, OPT_BASIS);
        parser.addOption('r', "ray-samples", "Number of ray sample points", true, OPT_RAYSAMPLES);
        parser.addOption('m', "max-stepsize", "Maximum step size to use in function-space approach. 0 for infinite, -1 for automatic", true, OPT_MAX_STEPSIZE);
        parser.addOption('\0', "beckmann", "Sample normals using beckmman distribution", false, OPT_BECKMANN);
        parser.addOption('\0', "weight-space", "Sample normals using weight space approach", false, OPT_WEIGHTSPACE);
        parser.addOption('\0', "function-space", "Sample normals using function space approach", false, OPT_FUNCTIONSPACE);
        parser.addOption('\0', "tag", "Additional tag to append to output file", true, OPT_TAG);
        parser.addOption('\0', "bound", "Distance from surface to start tracing. -1 for automatic", true, OPT_BOUND);
        parser.addOption('\0', "solver-threshold", "Threshold below which Eigen discards singular values.", true, OPT_SOLVER_THRESHOLD);
        
        parser.parse(argc, argv);

        if (parser.operands().empty() || parser.isPresent(OPT_HELP)) {
            parser.printHelpText();
            std::exit(0);
        }

        int _threadCount;
        Path _inputDirectory; 
        Path _outputDirectory; 
        Path _outputFile; 
        int _spp = 100000;
        uint32 _seed = 0xBA5EBA11;
        double _angle = 0;
        int _numBasis = 300;
        int _numRaySamples = 64;
        double _maxStepSize = -1;
        double _bound = -1;
        double _solverThreshold = 0;
        std::string _tag;

        bool _doWeightspace = parser.isPresent(OPT_WEIGHTSPACE);
        bool _doBeckmann = parser.isPresent(OPT_BECKMANN);
        bool _doFunctionspace = parser.isPresent(OPT_FUNCTIONSPACE);

        if (parser.isPresent(OPT_THREADS)) {
            int newThreadCount = std::atoi(parser.param(OPT_THREADS).c_str());
            if (newThreadCount > 0)
                _threadCount = newThreadCount;
        }

        ThreadUtils::startThreads(_threadCount);

        if (parser.isPresent(OPT_INPUT_DIRECTORY)) {
            _inputDirectory = Path(parser.param(OPT_INPUT_DIRECTORY));
            _inputDirectory.freezeWorkingDirectory();
            _inputDirectory = _inputDirectory.absolute();
        }

        if (parser.isPresent(OPT_OUTPUT_DIRECTORY)) {
            _outputDirectory = Path(parser.param(OPT_OUTPUT_DIRECTORY));
            _outputDirectory.freezeWorkingDirectory();
            _outputDirectory = _outputDirectory.absolute();
            if (!_outputDirectory.exists())
                FileUtils::createDirectory(_outputDirectory, true);
        }

        if (parser.isPresent(OPT_SPP))
            _spp = std::atoi(parser.param(OPT_SPP).c_str());

        if (parser.isPresent(OPT_SEED))
            _seed = std::atoi(parser.param(OPT_SEED).c_str());

        if (parser.isPresent(OPT_ANGLE))
            _angle = std::atof(parser.param(OPT_ANGLE).c_str());

        if (parser.isPresent(OPT_BASIS))
            _numBasis = std::atoi(parser.param(OPT_BASIS).c_str());

        if (parser.isPresent(OPT_RAYSAMPLES))
            _numRaySamples = std::atoi(parser.param(OPT_RAYSAMPLES).c_str());

        if (parser.isPresent(OPT_MAX_STEPSIZE))
            _maxStepSize = std::atof(parser.param(OPT_MAX_STEPSIZE).c_str());

        if (parser.isPresent(OPT_BOUND))
            _bound = std::atof(parser.param(OPT_BOUND).c_str());

        if (parser.isPresent(OPT_TAG))
            _tag = parser.param(OPT_TAG);

        if (parser.isPresent(OPT_SOLVER_THRESHOLD))
            _solverThreshold = std::atof(parser.param(OPT_SOLVER_THRESHOLD).c_str());

        for (const std::string &p : parser.operands()) {
            std::shared_ptr<JsonDocument> document;
            try {
                document = std::make_shared<JsonDocument>(Path(p));
            }
            catch (std::exception& e) {
                std::cerr << e.what() << "\n";
                continue;
            }

            Scene scene;

            auto covs = (*document)["covariances"];
            if (!covs || !covs.isArray()) {
                std::cerr << "There should be a `covariances` array in the file\n";
                return -1;
            }

            auto lmean = std::make_shared<LinearMean>(Vec3d(0.f), Vec3d(0.f, 0.f, 1.f), 1.0f);

            for (int i = 0; i < covs.size(); i++) {
                const auto& jcov = covs[i];
                try {
                    auto cov = instantiate<CovarianceFunction>(jcov, scene);
                    cov->loadResources();

                    std::cout << "============================================\n";
                    std::cout << cov->id() << "\n";

                    auto gp = std::make_shared<GaussianProcess>(lmean, cov);
                    gp->_covEps = _solverThreshold;
                    gp->_maxEigenvaluesN = 1024;

                    float alpha = gp->_cov->compute_beckmann_roughness();

                    float bound = _bound;
                    if(_bound < 0) {
                        bound = gp->noIntersectBound(Vec3d(0.), 0.9999);
                    }

                    float maxStepsize = _maxStepSize;
                    if(_maxStepSize < 0) {
                        maxStepsize = gp->goodStepsize(Vec3d(0.), 0.99);
                    }

                    std::cout << "Beckmann roughness: " << alpha << "\n";
                    std::cout << "0.9999 percentile: " << bound << "\n";
                    std::cout << "Good stepsize: " << maxStepsize << "\n";


                    if(_doWeightspace) {
                        std::cout << "Weight space sampling...\n";
                        microfacet_intersect_test(gp, _outputDirectory, _spp, _numBasis, _angle, bound, _seed, _tag);
                    }

                    if(_doBeckmann) {
                        std::cout << "Beckmann sampling...\n";
                        microfacet_sample_beckmann(gp, _outputDirectory, _spp, _angle, bound, _seed, _tag);
                    }

                    if(_doFunctionspace) {
                        std::cout << "Function space sampling...\n";
                        ndf_cond_validate(gp, _spp, _seed, _outputDirectory, _angle, GPNormalSamplingMethod::ConditionedGaussian, bound, maxStepsize, _numRaySamples, _tag);
                    }

                } catch (std::exception& e) {
                    std::cerr << e.what() << "\n";
                }
            }
        }
    }
}
