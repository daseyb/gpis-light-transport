#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <integrators/path_tracer/PathTracer.hpp>
#include <thread/ThreadUtils.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

void record_first_hit(std::string scene_file, TraceableScene* tracableScene) {

    PathTracerSettings settings;
    settings.enableConsistencyChecks = false;
    settings.enableLightSampling = false;
    settings.enableTwoSidedShading = true;
    settings.enableVolumeLightSampling = false;
    settings.includeSurfaces = true;
    settings.lowOrderScattering = true;
    settings.maxBounces = 5;
    settings.minBounces = 0;

    PathTracer pathTracer(tracableScene, settings, 0);

    int samples = 1000000;
    Eigen::MatrixXf normals(samples, 3);
    Eigen::MatrixXf distanceSamples(samples, 1);
    Eigen::MatrixXf reflectionDirs(samples, 3);

    std::string scene_id = scene_file.find("ref") != std::string::npos ? "surface" : "medium";

    Path basePath = Path("testing/intersections") / Path(scene_file).baseName();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int sid = 0;

    Vec3f rdir;


    if (scene_id == "medium") {
        pathTracer._firstMediumBounceCb = [&sid, &normals, &reflectionDirs, &distanceSamples, &rdir](const MediumSample& mediumSample, Ray r) {
            rdir = r.dir();

            auto normal = mediumSample.aniso.normalized();
            normals(sid, 0) = normal.x();
            normals(sid, 1) = normal.z();
            normals(sid, 2) = normal.y();

            reflectionDirs(sid, 0) = r.dir().x();
            reflectionDirs(sid, 1) = r.dir().z();
            reflectionDirs(sid, 2) = r.dir().y();

            distanceSamples(sid, 0) = mediumSample.t;

            return false;
        };
    }
    else {
        pathTracer._firstSurfaceBounceCb = [&sid, &normals, &reflectionDirs](const SurfaceScatterEvent& event, Ray r) {
            auto normal = event.info->Ns;
            normals(sid, 0) = normal.x();
            normals(sid, 1) = normal.z();
            normals(sid, 2) = normal.y();

            reflectionDirs(sid, 0) = r.dir().x();
            reflectionDirs(sid, 1) = r.dir().z();
            reflectionDirs(sid, 2) = r.dir().y();

            return false;
        };
    }


    UniformPathSampler sampler(0);
    sampler.next2D();

    for (sid = 0; sid < samples; sid++) {
        if ((sid + 1) % 100 == 0) {
            std::cout << sid << "/" << samples;
            std::cout << "\r";
        }

        pathTracer.traceSample({ 256, 256 }, sampler);
    }

    std::cout << acos(rdir.normalized().y()) / PI * 180 << "\n";

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-normals.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)normals.data(), sizeof(float) * normals.rows() * normals.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-reflection.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)reflectionDirs.data(), sizeof(float) * reflectionDirs.rows() * reflectionDirs.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-distances.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)distanceSamples.data(), sizeof(float) * distanceSamples.rows() * distanceSamples.cols());
        xfile.close();
    }
}

void first_intersect_ansio(std::string scene_file, Scene* scene) {

    TraceableScene* tracableScene = scene->makeTraceable();


    PathTracerSettings settings;
    settings.enableConsistencyChecks = false;
    settings.enableLightSampling = false;
    settings.enableTwoSidedShading = true;
    settings.enableVolumeLightSampling = false;
    settings.includeSurfaces = true;
    settings.lowOrderScattering = true;
    settings.maxBounces = 8;
    settings.minBounces = 0;

    PathTracer pathTracer(tracableScene, settings, 0);

    Path basePath = Path("testing/intersections") / Path(scene_file).baseName();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }
    
    std::vector<Vec2u> pixelpositions;
    std::vector<Eigen::Matrix3f> anisotropies;
    std::vector<Vec3f> intersectPositions;

    auto cov = std::static_pointer_cast<MeanGradNonstationaryCovariance>( 
        std::static_pointer_cast<GaussianProcess>(
            std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0])->_gp)->_cov);

    Vec2u currentPixel;

    pathTracer._firstMediumBounceCb = [&currentPixel, &anisotropies, &pixelpositions, &cov, &intersectPositions](const MediumSample& mediumSample, Ray r) {
        pixelpositions.push_back(currentPixel);
        anisotropies.push_back(cov->localAniso(vec_conv<Vec3d>(r.pos())).cast<float>());
        intersectPositions.push_back(r.pos());
        return false;
    };

    pathTracer._firstSurfaceBounceCb = [](const SurfaceScatterEvent& event, Ray r) {
        return event.sampledLobe.isForward();
    };

    UniformPathSampler sampler(0);
    sampler.next2D();

    for (currentPixel.x() = 0; currentPixel.x() < 512; currentPixel.x()++) {
        if ((currentPixel.x() + 1) % 10 == 0) {
            std::cout << currentPixel.x() << "/" << 512;
            std::cout << "\r";
        }

        for (currentPixel.y() = 0; currentPixel.y() < 512; currentPixel.y()++) {
            pathTracer.traceSample(currentPixel, sampler);
        }
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/anisotropies.bin")),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)anisotropies.data(), sizeof(float) * anisotropies.size() * anisotropies[0].rows() * anisotropies[0].cols());
        xfile.close();
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/anisotropies-intersects.bin")),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)intersectPositions.data(), sizeof(Vec3f) * intersectPositions.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/anisotropies-pixelpos.bin")),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)pixelpositions.data(), sizeof(Vec2u) * pixelpositions.size());
        xfile.close();
    }
}


void record_paths(std::string scene_file, TraceableScene* tracableScene) {

    PathTracerSettings settings;
    settings.enableConsistencyChecks = false;
    settings.enableLightSampling = false;
    settings.enableTwoSidedShading = true;
    settings.enableVolumeLightSampling = false;
    settings.includeSurfaces = true;
    settings.lowOrderScattering = true;
    settings.maxBounces = 8;
    settings.minBounces = 0;

    PathTracer pathTracer(tracableScene, settings, 0);

    int samples = 1000;
    Eigen::MatrixXf path_points(samples * settings.maxBounces, 3);
    path_points.setZero();

    std::vector<Vec3f> normals(samples * settings.maxBounces);

    std::string scene_id = scene_file.find("ref") != std::string::npos ? "surface" : "medium";

    Path basePath = Path("testing/intersections") / Path(scene_file).baseName();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int sid = 0;
    int bounce = 0;

    pathTracer._firstMediumBounceCb = [&sid, &path_points, &normals, &bounce](const MediumSample& mediumSample, Ray r) {
        path_points.row(sid * 8 + bounce) =  vec_conv<Eigen::Vector3f>(r.pos());
        normals[sid * 8 + bounce] = vec_conv<Vec3f>(mediumSample.aniso.normalized());
        bounce++;
        return true;
    };
    pathTracer._firstSurfaceBounceCb = [&sid, &path_points, &normals, &bounce](const SurfaceScatterEvent& event, Ray r) {
        if (!event.sampledLobe.isForward()) {
            path_points.row(sid * 8 + bounce) = vec_conv<Eigen::Vector3f>(r.pos());
            normals[sid * 8 + bounce] = event.info->Ns;
            bounce++;
        }
        return true;
    };


    UniformPathSampler sampler(0);
    sampler.next2D();

    for (sid = 0; sid < samples; sid++) {
        if ((sid + 1) % 100 == 0) {
            std::cout << sid << "/" << samples;
            std::cout << "\r";
        }

        path_points.row(sid * 8) = vec_conv<Eigen::Vector3f>(tracableScene->cam().pos());
        normals[sid * 8] = Vec3f(1.f);
        bounce = 1;
        pathTracer.traceSample({ 256, 256 }, sampler);
        //pathTracer.traceSample({ 256, 187 }, sampler);
        //pathTracer.traceSample({ 180, 180 }, sampler);
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-paths.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)path_points.data(), sizeof(float) * path_points.rows() * path_points.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-paths-normals.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)normals.data(), sizeof(Vec3f) * normals.size());
        xfile.close();
    }
}
 
int main(int argc, char** argv) {
    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    for (int arg = 1; arg < argc; arg++) {
        Scene* scene = nullptr;
        try {
            scene = Scene::load(Path(argv[arg]));
            scene->loadResources();
        }
        catch (std::exception& e) {
            std::cout << e.what();
            return -1;
        }

        auto tracableScene = scene->makeTraceable();

        //first_intersect_ansio(argv[arg], scene);

        //record_first_hit(argv[arg], tracableScene);

        record_paths(argv[arg], tracableScene);
    }
}
