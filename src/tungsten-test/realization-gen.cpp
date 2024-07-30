#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>
#include <thread/ThreadUtils.hpp>
#include <io/Scene.hpp>
#include <math/WeightSpaceGaussianProcess.hpp>
#include <ccomplex>
#include <fftw3.h>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 32;

std::tuple<std::vector<Vec3d>, std::vector<Vec3d>, std::vector<double>, std::vector<Derivative>> sample_surface(float scale) {

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ns;
    std::vector<double> vs;
    std::vector<Derivative> ds;

    Vec3d c = Vec3d(0.75, 0, 0.f);
    double r = 0.4f;
    /*ps.push_back(c);
    ns.push_back(Vec3d(0.f));
    vs.push_back(-r);
    ds.push_back(Derivative::None);

   ps.push_back(c + r * 2 * Vec3d(-1.f, 1.f, 0.f).normalized());
    ns.push_back(Vec3d(0.f));
    vs.push_back(r);
    ds.push_back(Derivative::None);*/

    int num_pts = 5;
    for (int i = 0; i < num_pts; i++) {
        float a = lerp(PI/2, PI * 1.5f, float(i) / (num_pts-1));
        Vec3d p = c + Vec3d(cos(a), sin(a), 0.f) * r;
        ps.push_back(p*scale);
        ns.push_back((p - c).normalized());
        vs.push_back(0);
        ds.push_back(Derivative::None);

#if 1
        ps.push_back(p * scale);
        ns.push_back((p - c).normalized());
        vs.push_back(1);
        ds.push_back(Derivative::First);
#endif
    }

    return { ps, ns, vs, ds };
}

void gen_cond_test() {

    int num_reals = 2000;
    
    float scale = 10;
    auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(scale, 0.3f * scale));

    std::cout << gp->compute_beckmann_roughness(Vec3d(0.)) << "\n";

    //return;

#if 0
    {
        UniformPathSampler sampler(0);
        sampler.next1D();
        sampler.next1D();


        std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
        std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    points[idx][2] = 0.f;
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }

            Eigen::MatrixXf samples = gp.sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3f(1.0f, 0.0f, 0.0f), num_reals, sampler);

            {
                std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-grid-samples-nocond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
#endif

    auto [cps, cns, cvs, cds] = sample_surface(10.f);

    gp->setConditioning(
        cps,
        cds,
        cns,
        cvs
    );

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-ps-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondPs.data(), sizeof(Vec3d) * gp->_globalCondPs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-ds-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondDerivs.data(), sizeof(Derivative) * gp->_globalCondDerivs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-ns-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondDerivDirs.data(), sizeof(Vec3d) * gp->_globalCondDerivDirs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-vs-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
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
                    points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f) * scale;
                    points[idx][2] = 0.f;
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }

            std::array<Vec3d, 2> ad_cond = {
                Vec3d(0.),
                Vec3d(-0.5f, 0, 0.)
            };

            std::array<Derivative, 2> add_deriv = {
                Derivative::None,
                Derivative::None,
            };

            auto add_v = std::make_shared<GPRealNodeValues>(-0.1 * Eigen::MatrixXd::Ones(2, 1), gp.get());


            auto [samples, gpIds] = gp->sample_cond(
                points.data(), derivs.data(), points.size(), nullptr,
                ad_cond.data(), add_v.get(), add_deriv.data(), 0, nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), num_reals, sampler)->flatten();

            {
                std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-grid-samples-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
}

void gen_real_microfacet_to_volume() {

    UniformPathSampler sampler(0);
    sampler.next2D();


    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                points[idx] = 10. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                points[idx][2] = 0.f;
                derivs[idx] = Derivative::None;
                idx++;
            }
        }
    }

    for (auto [meanScale, meanMin] : { std::tuple{1.f, -10000.f}, std::tuple{5.f, -100.f}, std::tuple{100.f, 1.f} }) {
        for (auto [isotropy, lengthScale] : { std::tuple{0.05f, 0.2f}, std::tuple{0.1f, 0.1f}, std::tuple{1.f, 0.05f} }) {
            UniformPathSampler sampler(0);
            sampler.next2D();

            auto gp = std::make_shared<GaussianProcess>(
                std::make_shared<LinearMean>(Vec3d(0., 0., 0.), Vec3d(0., 1., 0.), meanScale, meanMin),
                std::make_shared<SquaredExponentialCovariance>(1.0f, lengthScale, Vec3f(1.f, isotropy, 1.f)));

            auto [samples, gpIds] = gp->sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), 1, sampler)->flatten();

            auto path = Path(tinyformat::format("testing/realizations/volume-to-surface/%s-%f-%d-grid-samples-cond.bin", gp->_cov->id(), meanScale, NUM_SAMPLE_POINTS));
            
            if (!path.parent().exists()) {
                FileUtils::createDirectory(path.parent());
            }

            {
                std::ofstream xfile(path.asString(), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }


}

void sample_scene_gp(int argc, const char** argv) {

    int dim = 32;

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.asString() << "\n";

    Scene* scene = nullptr;
    TraceableScene* tscene = nullptr;
    try {
        scene = Scene::load(scenePath);
        scene->loadResources();
        tscene = scene->makeTraceable();
    }
    catch (std::exception& e) {
        std::cout << e.what();
        return;
    }

    std::shared_ptr<GaussianProcessMedium> gp_medium = std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0]);

    auto gp = std::static_pointer_cast<GPSampleNode>(gp_medium->_gp);

    UniformPathSampler sampler(0);
    sampler.next2D();

    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim);
    std::vector<Derivative> fderivs(dim * dim);

    auto processBox = scene->findPrimitive("processBox");

    Vec3d min = vec_conv<Vec3d>(processBox->bounds().min());
    Vec3d max = vec_conv<Vec3d>(processBox->bounds().max());

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>((max - min) / dim));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(min));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));

    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                int k = dim / 2;
                int idx = i * dim + j ;
                points[idx] = lerp(min, max, Vec3d((float)i, (float)j, (float)k) / (dim));
                derivs[idx] = Derivative::None;
            }
        }
    }

    auto [samples, gpIds] = gp->sample(
        points.data(), derivs.data(), points.size(), nullptr,
        nullptr, 0,
        Vec3d(0.0f, 0.0f, 0.0f), 1, sampler)->flatten();

    auto mean = gp->mean(points.data(), derivs.data(), nullptr, Vec3d(0.), points.size());


    auto path = Path(tinyformat::format("testing/realizations/scene/%s/%s-%d-samples.bin", 
        scenePath.parent().baseName().asString(), 
        scenePath.baseName().stripExtension().asString(), dim));

    if (!path.parent().exists()) {
        FileUtils::createDirectory(path.parent());
    }

    {
        std::ofstream xfile(path.asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/scene/%s/%s-%d-mean.bin",
            scenePath.parent().baseName().asString(),
            scenePath.baseName().stripExtension().asString(), dim), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(double) * mean.rows() * mean.cols());
        xfile.close();
    }
}

void gen__didactic_fig_reals() {

    UniformPathSampler sampler(0);
    sampler.next2D();

    int dim = 64;

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<ProceduralMean>(SdfFunctions::Function::TwoSpheres), std::make_shared<SquaredExponentialCovariance>(0.2f, 0.1f));
    
    const WeightSpaceRealization ws = WeightSpaceBasis::sample(gp->_cov, 300, sampler).sampleRealization(gp, sampler);

    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim, Derivative::None);

    {
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (dim - 1) - 0.5f);
                points[idx][2] = 0.f;
                idx++;
            }
        }
    }

    auto sample = ws.evaluate(points.data(), points.size());

    Path basePath = Path("testing/diadactic-fig");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    {
        std::ofstream xfile(basePath.asString() + "/global-real.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
        xfile.close();
    }


    auto desired_intersect_pts = std::array<Vec3d, 5>{
        Vec3d(0.2291, 1.1383, 0.),
        Vec3d(0.8282, 0.8913, 0.),
        Vec3d(0.9453, 1.3886, 0.),
        Vec3d(1.3047, 0.6843, 0.),
        Vec3d(1.8209, 0.8000, 0.)
    };

    for (auto& p : desired_intersect_pts) {
        Vec3d np = Vec3d(0.);
        np.x() = p.x() - 1;
        np.y() = (p.y() - 1) * -1;
        p = np;
    }

    std::vector<Vec3d> interp_points;
    std::vector<double> interp_ts;
    double startT = 0;

    {
        auto mean = gp->mean_prior(points.data(), derivs.data(), nullptr, Vec3d(), points.size());
        std::ofstream xfile(basePath.asString() + "/mean.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(mean[0]) * mean.size());
        xfile.close();
    }

    
    for (int i = 1; i < desired_intersect_pts.size(); i++) {
        Vec3d pp = desired_intersect_pts[i - 1];
        Vec3d cp = desired_intersect_pts[i];


        int path_res = 32;
        for (int j = 1; j <= path_res; j++) {
            interp_points.push_back(lerp(pp, cp, double(j) / (path_res)));
            interp_ts.push_back(startT + (interp_points.back() - pp).length());
        }

        startT = interp_ts.back();

        auto interp_p_values = ws.evaluate(interp_points.data(), interp_points.size());
        std::vector<Derivative> interp_derivs(interp_points.size(), Derivative::None);
        GPRealNodeValues cond_real(interp_p_values, gp.get());

        std::vector<Vec3d> interp_deriv_dirs(interp_points.size(), Vec3d());

        std::vector<double> interp_values(interp_p_values.data(), interp_p_values.data() + interp_p_values.size());

        {
            std::ofstream xfile(basePath.asString() + tinyformat::format("/ray-values-%d.bin", i), std::ios::out | std::ios::binary);
            xfile.write((char*)interp_values.data(), sizeof(interp_values[0]) * interp_values.size());
            xfile.close();
        }

        {
            std::ofstream xfile(basePath.asString() + tinyformat::format("/ray-ts-%d.bin", i), std::ios::out | std::ios::binary);
            xfile.write((char*)interp_ts.data(), sizeof(interp_ts[0]) * interp_ts.size());
            xfile.close();
        }

        continue;

        gp->setConditioning(interp_points, interp_derivs, interp_deriv_dirs, interp_values);

        /*auto [samples, gpIds] = gp->sample_cond(
            points.data(), derivs.data(), points.size(), nullptr,
            interp_points.data(), &cond_real, interp_derivs.data(), interp_points.size(), nullptr,
            nullptr, 0,
            Vec3d(0.0f, 0.0f, 0.0f), 1, sampler)->flatten();*/

        UniformPathSampler real_samp(0);
        real_samp.next2D();
        auto [samples, gpIds] = gp->sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3d(0.0f, 0.0f, 0.0f), 5000, real_samp)->flatten();


        {
            std::ofstream xfile(basePath.asString() + tinyformat::format("/local-real-%d.bin", i), std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(samples[0]) * samples.size());
            xfile.close();
        }

        std::cout << "Written path segment " << i << std::endl;
    }
}

void gen__mem_model_fig_reals() {
    UniformPathSampler sampler(0);
    sampler.next2D();

    int dim = 32;

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<LinearMean>(Vec3d(0., 0., 0.), Vec3d(0., 1., 0.), 1), std::make_shared<SquaredExponentialCovariance>(0.2f, 0.1f));

    const WeightSpaceRealization ws = WeightSpaceBasis::sample(gp->_cov, 300, sampler).sampleRealization(gp, sampler);

    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim, Derivative::None);

    {
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (dim - 1) - 0.5f);
                points[idx][2] = 0.f;
                idx++;
            }
        }
    }

    auto sample = ws.evaluate(points.data(), points.size());

    Path basePath = Path("testing/mem-model-fig");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }


    {
        std::ofstream xfile(basePath.asString() + "/global-real.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
        xfile.close();
    }

    auto desired_intersect_pts = std::array<Vec3d, 5>{
        Vec3d(0.289, 0.5602, 0.),
            Vec3d(0.8164, 1.0627, 0.),
    };

    for (auto& p : desired_intersect_pts) {
        Vec3d np = Vec3d(0.);
        np.x() = p.x() - 1;
        np.y() = (p.y() - 1) * -1;
        p = np;
    }

    std::vector<Vec3d> interp_points;
    std::vector<double> interp_ts;
    double startT = 0;

    Vec3d pp = desired_intersect_pts[0];
    Vec3d cp = desired_intersect_pts[1];

    int path_res = 32;
    for (int j = 0; j < path_res; j++) {
        interp_points.push_back(lerp(pp, cp, double(j) / (path_res)));
        interp_ts.push_back(startT + (interp_points.back() - pp).length());
    }


    auto interp_p_values = ws.evaluate(interp_points.data(), interp_points.size());
    auto intersect_normal = ws.evaluateGradient(interp_points.back());
    std::vector<Derivative> interp_derivs(interp_points.size(), Derivative::None);
    std::vector<Vec3d> interp_deriv_dirs(interp_points.size(), Vec3d());
    std::vector<double> interp_values(interp_p_values.data(), interp_p_values.data() + interp_p_values.size());


    {
        gp->setConditioning(interp_points, interp_derivs, interp_deriv_dirs, interp_values);

        UniformPathSampler real_samp(0);
        real_samp.next2D();
        auto [samples, gpIds] = gp->sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3d(0.0f, 0.0f, 0.0f), 200, real_samp)->flatten();

        {
            std::ofstream xfile(basePath.asString() + "/local-real-elephant.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(samples[0]) * samples.size());
            xfile.close();
        }
    }


    interp_points.push_back(interp_points.back());
    interp_points.push_back(interp_points.back());
    interp_points.push_back(interp_points.back());

    interp_derivs.push_back(Derivative::First);
    interp_derivs.push_back(Derivative::First);
    interp_derivs.push_back(Derivative::First);

    interp_deriv_dirs.push_back(Vec3d(1., 0., 0.));
    interp_deriv_dirs.push_back(Vec3d(0., 1., 0.));
    interp_deriv_dirs.push_back(Vec3d(0., 0., 1.));

    interp_values.push_back(intersect_normal.x());
    interp_values.push_back(intersect_normal.y());
    interp_values.push_back(intersect_normal.z());

    {
        interp_points.erase(interp_points.begin(), interp_points.end() - 4);
        interp_derivs.erase(interp_derivs.begin(), interp_derivs.end() - 4);
        interp_deriv_dirs.erase(interp_deriv_dirs.begin(), interp_deriv_dirs.end() - 4);
        interp_values.erase(interp_values.begin(), interp_values.end() - 4);
        gp->setConditioning(interp_points, interp_derivs, interp_deriv_dirs, interp_values);

        UniformPathSampler real_samp(0);
        real_samp.next2D();
        auto [samples, gpIds] = gp->sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3d(0.0f, 0.0f, 0.0f), 200, real_samp)->flatten();

        {
            std::ofstream xfile(basePath.asString() + "/local-real-goldfish.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(samples[0]) * samples.size());
            xfile.close();
        }
    }

    {
        gp->setConditioning({ {interp_points.back()} }, { {Derivative::First} }, { {Vec3d(0.)} }, { {interp_p_values[interp_p_values.size()-1]} });

        UniformPathSampler real_samp(0);
        real_samp.next2D();
        auto [samples, gpIds] = gp->sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3d(0.0f, 0.0f, 0.0f), 200, real_samp)->flatten();

        {
            std::ofstream xfile(basePath.asString() + "/local-real-dori.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(samples[0]) * samples.size());
            xfile.close();
        }

    }
}

void gen__ws_limited_reals() {

    auto gen_for_cov = [](std::shared_ptr<CovarianceFunction> cov) {
        Path basePath = Path("testing/ws-limited-basis-fig/" + cov->id());
        if (!basePath.exists()) {
            FileUtils::createDirectory(basePath);
        }

        int dim = 512;


        auto gp = std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 0.2f), cov);

        std::vector<Vec3d> points(dim * dim);
        std::vector<Derivative> derivs(dim * dim, Derivative::None);

        {
            int idx = 0;
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (dim - 1) - 0.5f);
                    points[idx][2] = 0.f;
                    idx++;
                }
            }
        }

        UniformPathSampler sampler(0);
        sampler.next2D();
        const WeightSpaceRealization ws = WeightSpaceBasis::sample(gp->_cov, 10000, sampler, false).sampleRealization(gp, sampler);

        for(int n : {10,100,1000,10000})
        {
            auto ws_trunc = ws.truncate(n);
            auto sample = ws_trunc.evaluate(points.data(), points.size());
            {
                std::ofstream xfile(basePath.asString() + tinyformat::format("/ws-real-%d.bin", n), std::ios::out | std::ios::binary);
                xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
                xfile.close();
            }
        }


        /* {
            UniformPathSampler real_samp(0);
            real_samp.next2D();
            auto [samples, gpIds] = gp->sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), 1, real_samp)->flatten();

            {
                std::ofstream xfile(basePath.asString() + "/fs-real.bin", std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(samples[0]) * samples.size());
                xfile.close();
            }

        }*/
        
    };

    gen_for_cov(std::make_shared<SquaredExponentialCovariance>(0.2f, 0.1f));
    gen_for_cov(std::make_shared<RationalQuadraticCovariance>(0.2f, 0.1f, 0.1f));
}



bool intersectRayAAWriteData(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, const Ray& ray, Vec3d& p, Path basePath) {
    const double sig_0 = (ray.farT() - ray.nearT()) * 0.1f;
    const double delta = 0.001;
    const double np = 1.5;
    const double nm = 0.5;

    double t = 0;
    double sig = sig_0;

    auto rd = vec_conv<Vec3d>(ray.dir());

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;

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

        t += max(nsig * 0.98, delta);

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



void gen__ws_algo_vis() {
    auto cov = std::make_shared<SquaredExponentialCovariance>(0.2f, 0.1f);

    Path basePath = Path("testing/ws-algo-vis-fig");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int dim = 64;


    auto gp = std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(1., 0., 0.), 0.2f), cov);

    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim, Derivative::None);

    {
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (dim - 1) - 0.5f);
                points[idx][2] = 0.f;
                idx++;
            }
        }
    }

    UniformPathSampler sampler(0);
    sampler.next2D();
    auto basis = WeightSpaceBasis::sample(gp->_cov, 100, sampler);
    const WeightSpaceRealization ws = basis.sampleRealization(gp, sampler);
    {
        auto sample = ws.evaluate(points.data(), points.size());
        std::ofstream xfile(basePath.asString() + tinyformat::format("/ws-real-%d.bin", 100), std::ios::out | std::ios::binary);
        xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
        xfile.close();
    }

    {
        auto mean = gp->mean_prior(points.data(), derivs.data(), nullptr, Vec3d(0.), points.size());
        std::ofstream xfile(basePath.asString() + tinyformat::format("/ws-mean.bin"), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(mean[0]) * mean.size());
        xfile.close();
    }

    for (int i = 0; i < 4; i++) {
        Eigen::VectorXd weights = Eigen::VectorXd::Zero(100);
        weights[i] = 1;

        Eigen::VectorXd result(points.size());
        for (size_t p = 0; p < points.size(); p++) {
            result[p] = basis.evaluate(vec_conv<Eigen::Vector3d>(points[p]), weights);
        }

        {
            std::ofstream xfile(basePath.asString() + tinyformat::format("/ws-basis-%d.bin", i), std::ios::out | std::ios::binary);
            xfile.write((char*)result.data(), sizeof(result[0]) * result.size());
            xfile.close();
        }
    }

    Vec3d ip;
    intersectRayAAWriteData(gp, ws, Ray(Vec3f(-0.8f, 0.f, 0.f), Vec3f(1.f, 0.f, 0.f), 0.f, 10.f), ip, basePath);
}


void gen__fs_algo_vis() {
    auto cov = std::make_shared<SquaredExponentialCovariance>(0.2f, 0.1f);

    Path basePath = Path("testing/fs-algo-vis-fig");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int dim = 500;


    auto gp = std::make_shared<GaussianProcess>(std::make_shared<LinearMean>(Vec3d(0., -0.8, 0.), Vec3d(0., 1., 0.)), cov);

    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim, Derivative::None);

    {
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (dim - 1) - 0.5f);
                points[idx][2] = 0.f;
                idx++;
            }
        }
    }

    UniformPathSampler sampler(0);
    sampler.next2D();
    auto basis = WeightSpaceBasis::sample(gp->_cov, 400, sampler);
    const WeightSpaceRealization ws = basis.sampleRealization(gp, sampler);
    {
        auto sample = ws.evaluate(points.data(), points.size());
        std::ofstream xfile(basePath.asString() + tinyformat::format("/ws-real-%d.bin", 400), std::ios::out | std::ios::binary);
        xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
        xfile.close();
    }

    auto desired_intersect_pts = std::array<Vec3d, 5>{
        Vec3d(0.8782, 1.0276, 0.),
            Vec3d(1.5708, 1.6521, 0.),
    };

    for (auto& p : desired_intersect_pts) {
        Vec3d np = Vec3d(0.);
        np.x() = p.x() - 1;
        np.y() = (p.y() - 1) * -1;
        p = np;
    }

    std::vector<Vec3d> interp_points;
    std::vector<double> interp_ts;
    double startT = 0;

    Vec3d pp = desired_intersect_pts[0];
    Vec3d cp = desired_intersect_pts[1];

    int path_res = 12;
    for (int j = 0; j < path_res; j++) {
        interp_points.push_back(lerp(pp, cp, double(j) / (path_res-1)));
        interp_ts.push_back(startT + (interp_points.back() - pp).length());
    }

    auto interp_p_values = ws.evaluate(interp_points.data(), interp_points.size());
    auto intersect_normal = ws.evaluateGradient(interp_points.back());
    std::vector<Derivative> interp_derivs(interp_points.size(), Derivative::None);
    std::vector<Vec3d> interp_deriv_dirs(interp_points.size(), Vec3d());
    std::vector<double> interp_values(interp_p_values.data(), interp_p_values.data() + interp_p_values.size());

    {
        std::ofstream xfile(basePath.asString() + tinyformat::format("/ray-values.bin"), std::ios::out | std::ios::binary);
        xfile.write((char*)interp_values.data(), sizeof(interp_values[0]) * interp_values.size());
        xfile.close();
    }

    {
        std::ofstream xfile(basePath.asString() + tinyformat::format("/ray-ts.bin"), std::ios::out | std::ios::binary);
        xfile.write((char*)interp_ts.data(), sizeof(interp_ts[0]) * interp_ts.size());
        xfile.close();
    }

    {
        auto mean = gp->mean_prior(points.data(), derivs.data(), nullptr, Vec3d(0.), points.size());
        std::ofstream xfile(basePath.asString() + tinyformat::format("/fs-mean.bin"), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(mean[0]) * mean.size());
        xfile.close();
    }


    {
        std::vector<double> probIntersect;
        for (const auto& p : points) {
            probIntersect.push_back(gp->cdf(p));
        }

        {
            std::ofstream xfile(basePath.asString() + tinyformat::format("/cdf.bin"), std::ios::out | std::ios::binary);
            xfile.write((char*)probIntersect.data(), sizeof(probIntersect[0]) * probIntersect.size());
            xfile.close();
        }
    }

    {
        gp->setConditioning(interp_points, interp_derivs, interp_deriv_dirs, interp_values);
        std::vector<double> probIntersect;
        for (const auto& p : points) {
            probIntersect.push_back(gp->cdf(p));
        }

        {
            std::ofstream xfile(basePath.asString() + tinyformat::format("/cdf-cond.bin"), std::ios::out | std::ios::binary);
            xfile.write((char*)probIntersect.data(), sizeof(probIntersect[0]) * probIntersect.size());
            xfile.close();
        }
    }


}

void gen__spectral_density_est() {

    auto cov = SquaredExponentialCovariance(1.0f, .25f);

    Path basePath = Path("testing/spectral-density-est/" + cov.id());
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }


    double max_t = 10;
    std::vector<double> covValues(pow(2, 12));
    std::vector<double> specValues(pow(2, 12));

    size_t i;
    covValues[0] = cov(Derivative::None, Derivative::None, Vec3d(0.), Vec3d(0.), Vec3d(), Vec3d());
    specValues[0] = cov.spectral_density(0);
    for (i = 1; i < covValues.size(); i++) {
        double t = double(i) / covValues.size() * max_t;
        covValues[i] = cov(Derivative::None, Derivative::None, Vec3d(0.), Vec3d(t, 0., 0.), Vec3d(), Vec3d()) / covValues[0];
        specValues[i] = cov.spectral_density(t*10);
    }

    auto dt = max_t / covValues.size();
    auto n = covValues.size();
    auto nfft = 2 * n - 2;
    auto nf = nfft / 2;

    // This is based on the pywafo tospecdata function: https://github.com/wafo-project/pywafo/blob/master/src/wafo/covariance/core.py#L163
    std::vector<double> acf;
    acf.insert(acf.end(), covValues.begin(), covValues.end());
    acf.insert(acf.end(), nfft - 2 * n + 2, 0.);
    acf.insert(acf.end(), covValues.rbegin() + 2, covValues.rend());

    std::vector<std::complex<double>> spectrumValues(acf.size());
    fftw_plan plan = fftw_plan_dft_r2c_1d(acf.size(), acf.data(), (fftw_complex*)spectrumValues.data(), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    std::vector<double> r_per;
    std::transform(spectrumValues.begin(), spectrumValues.end(), std::back_inserter(r_per), [](auto spec) { return std::max(spec.real(), 0.); });

    auto discreteSpectralDensity = std::vector<double>();
    std::transform(r_per.begin(), r_per.begin() + nf + 1, std::back_inserter(discreteSpectralDensity), [dt](auto per) { return std::abs(per) * dt / PI; });

    {
        std::ofstream xfile(basePath.asString() + tinyformat::format("/discrete-spectral-density.bin"), std::ios::out | std::ios::binary);
        xfile.write((char*)discreteSpectralDensity.data(), sizeof(discreteSpectralDensity[0]) * discreteSpectralDensity.size());
        xfile.close();
    }
    {
        std::ofstream xfile(basePath.asString() + tinyformat::format("/analytic-spectral-density.bin"), std::ios::out | std::ios::binary);
        xfile.write((char*)specValues.data(), sizeof(specValues[0]) * specValues.size());
        xfile.close();
    }

}

int main(int argc, const char** argv) {
    //gen__mem_model_fig_reals();
    //gen__ws_algo_vis();
    //gen__ws_limited_reals();
    gen__fs_algo_vis();

    //gen__spectral_density_est();
}
