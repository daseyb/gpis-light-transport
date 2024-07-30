#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <thread/ThreadUtils.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

int gen3d(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

    std::string prefix = "csg-two-spheres-nofilter";

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    Scene* scene = nullptr;
    TraceableScene* tscene = nullptr;
    try {
        scene = Scene::load(scenePath);
        scene->loadResources();
        tscene = scene->makeTraceable();
    }
    catch (std::exception& e) {
        std::cout << e.what();
        return -1;
    }

    std::shared_ptr<GaussianProcessMedium> gp_medium = std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0]);

    auto gp = std::static_pointer_cast<GPSampleNode>(gp_medium->_gp);

    UniformPathSampler sampler(0);
    sampler.next2D();

    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> fderivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    auto processBox = scene->findPrimitive("processBox");

    Vec3d min = vec_conv<Vec3d>(processBox->bounds().min());
    Vec3d max = vec_conv<Vec3d>(processBox->bounds().max());

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>((max - min) / NUM_SAMPLE_POINTS));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(min));

    Eigen::VectorXf mean(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf variance(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));

    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    int numEstSamples = 1;
    {
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
#pragma omp parallel for
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    points[idx] = lerp(min, max, Vec3d((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS));
                    derivs[idx] = Derivative::None;
                    fderivs[idx] = Derivative::First;

                    Eigen::VectorXd samples(numEstSamples);
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec3d offset = (Vec3d(sampler.next1D(), sampler.next1D(), sampler.next1D()) - 0.5) * 0.;
                        Vec3d p = lerp(min, max, (Vec3d((float)i, (float)j, (float)k) + offset) / (NUM_SAMPLE_POINTS));
                        auto [samp, gpidx] = gp->sample(&p, &derivs[idx], 1, nullptr, nullptr, 0, Vec3d(), 1, sampler)->flatten();
                        samples[s] = samp(0, 0);
                    }

                    mean[idx] = samples.mean();
                    variance[idx] = sqrt((samples.array() - mean[idx]).square().sum() / (samples.size() - 1));
                }
            }
        }

        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    meanAccessor.setValue({ i,j,k }, mean[idx]);
                }
            }
        }
    }

    {
        std::ofstream xfile(tinyformat::format("./testing/bake-csg/%s-mean-eval-avg-%d.bin", prefix, NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("./testing/bake-csg/%s-var-eval-avg-%d.bin", prefix, NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)variance.data(), sizeof(float) * variance.rows() * variance.cols());
        xfile.close();
    }


    {
        auto varGrid = openvdb::createGrid<openvdb::FloatGrid>(1.f);
        varGrid->setGridClass(openvdb::GRID_LEVEL_SET);
        varGrid->setName("variance");
        varGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));

        openvdb::FloatGrid::Accessor varAccessor = varGrid->getAccessor();


        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    varAccessor.setValue({ i,j,k }, variance[idx]);
                }
            }
        }

        {
            openvdb::GridPtrVec grids;
            grids.push_back(meanGrid);
            grids.push_back(varGrid);
            openvdb::io::File file(scenePath.parent().asString() + tinyformat::format("%s-eval-avg-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }

        /* {
            openvdb::GridPtrVec grids;
            grids.push_back(varGrid);
            openvdb::io::File file(tinyformat::format("./testing/bake-csg/%s-var-eval-avg-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }*/
    }

    /*{
        Eigen::VectorXf gradx = gp->mean(points.data(), fderivs.data(), nullptr, Vec3f(1.0f, 0.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dx-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)gradx.data(), sizeof(float) * gradx.rows() * gradx.cols());
        xfile.close();
    }
    {
        Eigen::VectorXf grady = gp->mean(points.data(), fderivs.data(), nullptr, Vec3f(0.0f, 1.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dy-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)grady.data(), sizeof(float) * grady.rows() * grady.cols());
        xfile.close();
    }
    {
        Eigen::VectorXf gradz = gp->mean(points.data(), fderivs.data(), nullptr, Vec3f(0.0f, 0.0f, 1.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dz-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)gradz.data(), sizeof(float) * gradz.rows() * gradz.cols());
        xfile.close();
    }*/


    return 0;

}

int test2d(int argc, char** argv) {
    return 0;
}

int main(int argc, char** argv) {

    return gen3d(argc, argv);
    //return test2d(argc, argv);

}
