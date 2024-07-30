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


void bsdf_sample(std::shared_ptr<GaussianProcess> gp, std::shared_ptr<PhaseFunction> phase, std::string phase_name, int samples, int seed, Path outputDir, float angle = (2 * PI) / 8, GPNormalSamplingMethod nsm = GPNormalSamplingMethod::ConditionedGaussian, float zrange = 4.f, float cov_step_size = 0.99f, int numRaySamplePoints = 64, std::string tag = "") {

    Path basePath = outputDir / Path("function-space") / Path(phase_name) / Path(gp->_cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    if (nsm == GPNormalSamplingMethod::Beckmann) {
        cov_step_size = 0.f;
    }

    auto gp_med = std::make_shared<FunctionSpaceGaussianProcessMedium>(
        gp, std::vector<std::shared_ptr<PhaseFunction>>({ phase }), 0, 1, 1, numRaySamplePoints,
        nsm == GPNormalSamplingMethod::Beckmann ? GPCorrelationContext::Goldfish : GPCorrelationContext::Goldfish,
        nsm == GPNormalSamplingMethod::Beckmann ? GPIntersectMethod::Mean : GPIntersectMethod::GPDiscrete,
        nsm, cov_step_size);

    gp_med->loadResources();
    gp_med->prepareForRender();

    UniformPathSampler sampler(seed);
    sampler.next2D();

    std::vector<double> sampled_phase_cos(samples);

    Ray ray = Ray(Vec3f(0.f, 0.f, 500.f), Vec3f(sin(angle / 180 * PI), 0.f, -cos(angle / 180 * PI)));

    {
        Mat4f mat = Mat4f::rotAxis(Vec3f(0.f, 0.f, 1.0f), 45);
        ray.setDir(mat.transformVector(ray.dir()));
        ray.setNearT(-(ray.pos().z() - zrange) / ray.dir().z());
        ray.setFarT(-(ray.pos().z() + zrange) / ray.dir().z());
    }


    for (int s = 0; s < samples;) {
        Medium::MediumState state;
        state.reset();

        MediumSample sample;
        Ray tRay = ray;
        bool success = gp_med->sampleDistance(sampler, tRay, state, sample);

        if (!success) {
            continue;
        }

        PhaseSample ps;
        if (!sample.phase->sample(sampler, ray.dir(), sample, ps)) {
            continue;
        }

        sampled_phase_cos[s] = ps.w.dot(ray.dir());
        s++;
    }

    {
        std::ofstream xfile(
            (basePath + Path(tinyformat::format("/%.1fdeg-%d-%s-%.3f-%.3f-%sphase-%d.bin", angle, numRaySamplePoints, GaussianProcessMedium::normalSamplingMethodToString(nsm), zrange, cov_step_size, tag, seed))).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)sampled_phase_cos.data(), sizeof(sampled_phase_cos[0]) * sampled_phase_cos.size());
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

    static const int OPT_THREADS = 1;
    static const int OPT_ANGLE = 2;
    static const int OPT_HELP = 3;
    static const int OPT_BASIS = 4;
    static const int OPT_OUTPUT_DIRECTORY = 5;
    static const int OPT_SPP = 6;
    static const int OPT_SEED = 7;
    static const int OPT_WEIGHTSPACE = 9;
    static const int OPT_FUNCTIONSPACE = 10;
    static const int OPT_INPUT_DIRECTORY = 11;
    static const int OPT_RAYSAMPLES = 12;
    static const int OPT_MAX_STEPSIZE = 13;
    static const int OPT_TAG = 14;
    static const int OPT_BOUND = 15;
    static const int OPT_SOLVER_THRESHOLD = 16;
    static const int OPT_MEAN = 17;

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
    parser.addOption('c', "cov-step-size", "Covariance used to determine step size to use in function-space approach. 0 for infinite, -1 for automatic", true, OPT_MAX_STEPSIZE);
    parser.addOption('m', "mean", "Mean offset", true, OPT_MEAN);
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
    Path _outputDirectory = "testing/phase";
    Path _outputFile;
    int _spp = 100000;
    uint32 _seed = 1;
    double _angle = 0;
    int _numBasis = 300;
    int _numRaySamples = 64;
    double _covStepSize = -1;
    double _bound = -1;
    double _solverThreshold = 0;
    std::string _tag;
    double _meanOffset = 0;

    bool _doWeightspace = parser.isPresent(OPT_WEIGHTSPACE);
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
        _covStepSize = std::atof(parser.param(OPT_MAX_STEPSIZE).c_str());

    if (parser.isPresent(OPT_BOUND))
        _bound = std::atof(parser.param(OPT_BOUND).c_str());

    if (parser.isPresent(OPT_TAG))
        _tag = parser.param(OPT_TAG);

    if (parser.isPresent(OPT_SOLVER_THRESHOLD))
        _solverThreshold = std::atof(parser.param(OPT_SOLVER_THRESHOLD).c_str());

    if (parser.isPresent(OPT_MEAN))
        _meanOffset = std::atof(parser.param(OPT_MEAN).c_str());

    for (const std::string& p : parser.operands()) {
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

        auto phases = (*document)["phases"];
        if (!phases || !phases.isArray()) {
            std::cerr << "There should be a `phases` array in the file\n";
            return -1;
        }

        auto mean = std::make_shared<HomogeneousMean>(_meanOffset);

        for (int j = 0; j < phases.size(); j++) {
            const auto& jphase = phases[j];
            auto phase = instantiate<PhaseFunction>(jphase, scene);
            phase->loadResources();

            std::string phase_name = "phase";
            jphase.getField("name", phase_name);

            for (int i = 0; i < covs.size(); i++) {
                const auto& jcov = covs[i];
                try {
                    auto cov = instantiate<CovarianceFunction>(jcov, scene);
                    cov->loadResources();

                    std::cout << "============================================\n";
                    std::cout << cov->id() << "\n";

                    auto gp = std::make_shared<GaussianProcess>(mean, cov);
                    gp->_covEps = _solverThreshold;
                    gp->_maxEigenvaluesN = 1024;

                    float alpha = gp->_cov->compute_beckmann_roughness();

                    float bound = _bound;
                    if (_bound < 0) {
                        bound = gp->noIntersectBound(Vec3d(0.), 0.9999);
                    }

                    std::cout << "Beckmann roughness: " << alpha << "\n";
                    std::cout << "0.9999 percentile: " << bound << "\n";

                    if (_doFunctionspace) {
                        std::cout << "Function space sampling...\n";
                        bsdf_sample(gp, phase, phase_name, _spp, _seed, _outputDirectory, _angle, GPNormalSamplingMethod::ConditionedGaussian, bound, _covStepSize, _numRaySamples, _tag);
                    }

                }
                catch (std::exception& e) {
                    std::cerr << e.what() << "\n";
                }
            }
        }
    }
}
