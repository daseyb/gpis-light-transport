#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <thread/ThreadUtils.hpp>
#include <math/WeightSpaceGaussianProcess.hpp>
#include <ccomplex>
#include <fftw3.h>
#include <core/media/FunctionSpaceGaussianProcessMedium.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

#ifdef CERES_AVAILABLE
#include "ceres/ceres.h"
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

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

    Eigen::VectorXf mean(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf variance(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf aniso(6 * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    
    auto cellSize = (max - min) / NUM_SAMPLE_POINTS;
    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>(cellSize));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(min));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));

    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    int numEstSamples = 100;
    {
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
#pragma omp parallel for
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    points[idx] = lerp(min, max, Vec3d((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS - 1));
                    derivs[idx] = Derivative::None;
                    fderivs[idx] = Derivative::First;

                    mean[idx] = 0;
                    std::vector<float> samples(numEstSamples);
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d p = points[idx] + offset * cellSize;
                        samples[s] = gp->mean(&p, &derivs[idx], nullptr, Vec3d(1.0f, 0.0f, 0.0f), 1)(0);
                        mean[idx] += samples[s];
                    }

                    mean[idx] /= numEstSamples;
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


        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::QuadraticSampler> meanGridSampler(
            meanGrid->tree(), 
            meanGrid->transform());


        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
#pragma omp parallel for
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    Vec3d cp = Vec3d((float)i, (float)j, (float)k);

                    std::vector<float> samples(numEstSamples);
                    int valid_samples = 0;
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d p = cp + offset;

                        if (p.x() < 0 || p.y() < 0 || p.z() < 0 ||
                            p.x() > NUM_SAMPLE_POINTS-1 || p.y() > NUM_SAMPLE_POINTS - 1 || p.z() > NUM_SAMPLE_POINTS - 1) {
                            samples[s] = 0;
                        }
                        else {
                            float meanSample = meanGridSampler.isSample(openvdb::Vec3R(p.x(), p.y(), p.z()));
                            Vec3d bp = lerp(min, max, p / (NUM_SAMPLE_POINTS - 1));
                            samples[s] = gp->mean(&bp, &derivs[idx], nullptr, Vec3d(1.0f, 0.0f, 0.0f), 1)(0) - meanSample;
                            valid_samples++;
                        }
                    }

                    variance[idx] = 0;
                    if (valid_samples > 1) {
                        for (int s = 0; s < numEstSamples; s++) {
                            variance[idx] += sqr(samples[s]);
                        }
                        variance[idx] /= (valid_samples - 1);
                    }

                    {
                        Vec3d bp = lerp(min, max, cp / (NUM_SAMPLE_POINTS - 1));

                        Vec3d ps[] = {
                            bp,bp,bp
                        };

                        Derivative derivs[]{
                            Derivative::First, Derivative::First, Derivative::First
                        };

                        Vec3d ddirs[] = {
                            Vec3d(1.f, 0.f, 0.f),
                            Vec3d(0.f, 1.f, 0.f),
                            Vec3d(0.f, 0.f, 1.f),
                        };

                        auto gps = gp->mean(ps, derivs, ddirs, Vec3d(0.f), 3);
                        auto grad = vec_conv<Vec3f>(gps);
                        TangentFrame tf(grad);

                        Eigen::Matrix3f vmat;
                        vmat.col(0) = vec_conv<Eigen::Vector3f>(tf.tangent);
                        vmat.col(1) = vec_conv<Eigen::Vector3f>(tf.bitangent);
                        vmat.col(2) = vec_conv<Eigen::Vector3f>(tf.normal);


                        Eigen::Matrix3f smat = Eigen::Matrix3f::Identity();
                        smat.diagonal() = Eigen::Vector3f{ 1.f, 1.f, 5.f };

                        Eigen::Matrix3f mat = vmat * smat * vmat.transpose();

                        aniso[idx * 6 + 0] = mat(0, 0);
                        aniso[idx * 6 + 1] = mat(1, 1);
                        aniso[idx * 6 + 2] = mat(2, 2);

                        aniso[idx * 6 + 3] = mat(0, 1);
                        aniso[idx * 6 + 4] = mat(0, 2);
                        aniso[idx * 6 + 5] = mat(1, 2);
                    }
                }
            }
        }
    }

    {
        auto varGrid = openvdb::createGrid<openvdb::FloatGrid>();
        openvdb::FloatGrid::Accessor varAccessor = varGrid->getAccessor();
        varGrid->setName("density");
        varGrid->setTransform(openvdb::math::Transform::createLinearTransform(4.0 / NUM_SAMPLE_POINTS));
        
        openvdb::GridPtrVec anisoGrids;
        std::vector<openvdb::FloatGrid::Accessor> anisoAccessors;

        std::string names[] = {
            "sigma_xx", "sigma_yy", "sigma_zz",
            "sigma_xy", "sigma_xz", "sigma_yz",
        };

        for (int i = 0; i < 6; i++) {
            auto anisoGrid = openvdb::createGrid<openvdb::FloatGrid>();
            anisoGrid->setName(names[i]);
            anisoGrid->setTransform(openvdb::math::Transform::createLinearTransform(4.0 / NUM_SAMPLE_POINTS));
            anisoAccessors.push_back(anisoGrid->getAccessor());
            anisoGrids.push_back(anisoGrid);
        }


        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    varAccessor.setValue({ i,j,k }, variance(idx));

                    for (int anisoIdx = 0; anisoIdx < 6; anisoIdx++) {
                        anisoAccessors[anisoIdx].setValue({ i,j,k }, aniso[idx * 6 + anisoIdx]);
                    }

                    idx++;
                }
            }
        }

        {
            openvdb::GridPtrVec grids;
            grids.push_back(meanGrid);
            grids.push_back(varGrid);
            grids.insert(grids.end(), anisoGrids.begin(), anisoGrids.end());
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/%s-isotopric-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }
    }

    return 0;

}

int mesh_convert(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    MeshSdfMean mean(std::make_shared<Path>(argv[1]), true);
    mean.loadResources();

    int dim = std::stod(argv[2]);

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    auto bounds = mean.bounds();
    bounds.grow(bounds.diagonal().length() * 0.2);

    Vec3d minp = bounds.min();
    Vec3d maxp = bounds.max();

    Vec3d extends = (maxp - minp);

    double max_extend = extends.max();

    std::cout << extends << ":" << max_extend << "\n";

    Vec3d aspect = extends / max_extend;

    Vec3i dims = vec_conv<Vec3i>(aspect * dim);

    std::cout << aspect << ":" << dims << "\n";

    auto cellSize = max_extend / dim;

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>(Vec3d(cellSize)));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    UniformPathSampler sampler(0);
    sampler.next2D();

    int num_samples = 5;

    for (int i = 0; i < dims.x(); i++) {
        std::cout << i << "\r";
        for (int j = 0; j < dims.y(); j++) {
#pragma omp parallel for
            for (int k = 0; k < dims.z(); k++) {
                auto p = lerp(minp, maxp, Vec3d((float)i, (float)j, (float)k) / vec_conv<Vec3d>(dims));

                double m = 0;
                
                if (num_samples > 0) {
                    for (int s = 0; s < num_samples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d pt = p + offset * cellSize * 0.1;
                        m += mean(Derivative::None, pt, Vec3d());
                    }
                    m /= num_samples;
                }
                else {
                    m = mean(Derivative::None, p, Vec3d());
                }

#pragma omp critical
                {
                    meanAccessor.setValue({ i,j,k }, m);
                }
            }
        }
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        openvdb::io::File file(scenePath.stripExtension().asString() + "-" + std::to_string(dim) + ".vdb");
        file.write(grids);
        file.close();
    }
}

int mesh_convert_2d(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    MeshSdfMean mean(std::make_shared<Path>(argv[1]), true);
    mean.loadResources();

    int dim = std::stod(argv[2]);

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    auto bounds = mean.bounds();
    bounds.grow(bounds.diagonal().length() * 0.2);

    Vec3d minp = bounds.min();
    Vec3d maxp = bounds.max();

    Vec3d extends = (maxp - minp);

    double max_extend = extends.max();

    std::cout << extends << ":" << max_extend << "\n";

    Vec3d aspect = extends / max_extend;

    Vec3i dims = vec_conv<Vec3i>(aspect * dim);

    std::cout << aspect << ":" << dims << "\n";

    auto cellSize = max_extend / dim;

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>(Vec3d(cellSize)));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    UniformPathSampler sampler(0);
    sampler.next2D();

    int num_samples = 1;

    for (int i = 0; i < dims.x(); i++) {
        std::cout << i << "\r";
#pragma omp parallel for
        for (int j = 0; j < dims.y(); j++) {
            auto p = lerp(minp, maxp, Vec3d((float)i, (float)j, (float)dims.z()/2) / vec_conv<Vec3d>(dims));

            double m = 0;

            if (num_samples > 0) {
                for (int s = 0; s < num_samples; s++) {
                    Vec2d s1 = rand_normal_2(sampler);
                    Vec2d s2 = rand_normal_2(sampler);
                    Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                    Vec3d pt = p + offset * cellSize * 0.1;
                    m += mean(Derivative::None, pt, Vec3d());
                }
                m /= num_samples;
            }
            else {
                m = mean(Derivative::None, p, Vec3d());
            }

#pragma omp critical
            {
                meanAccessor.setValue({ i,j, dims.z() / 2 }, m);
            }
        }
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        openvdb::io::File file(scenePath.stripExtension().asString() + "-" + std::to_string(dim) + "-2d.vdb");
        file.write(grids);
        file.close();
    }
}


std::vector<double> compute_acf_direct(const double* signal, size_t n, double mean) {

    std::vector<double> acf(n / 2);
    
    double var = 0;
    for (size_t i = 0; i < n; i++)
    {
        var += sqr(mean - signal[i]);
    }
    var /= n;

    for (size_t t = 0; t < acf.size(); t++)
    {
        double nu = 0; // Numerator
        double de = 0; // Denominator
        for (size_t i = 0; i < n; i++)
        {
            double xim = signal[i] - mean;
            nu += xim * (signal[(i + t) % n] - mean);
            de += xim * xim;
        }

        acf[t] = var * nu / de;
    }

    return acf;
}

std::vector<double> compute_acf_fftw_1D(const double* signal, size_t n, double mean) {
    // Allocate memory for the FFTW input and output arrays
    double* in = (double*)fftw_malloc(sizeof(double) * n);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1));

    // Create a plan for the forward FFT
    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);

    // Initialize the input array with the signal
    for (int i = 0; i < n; ++i) {
        in[i] = signal[i] - mean;
    }

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the power spectrum by squaring the magnitude of the FFT
    for (int i = 0; i < n / 2 + 1; ++i) {
        double magnitude = out[i][0] * out[i][0] + out[i][1] * out[i][1];
        out[i][0] = magnitude / n;
        out[i][1] = 0.0;
    }

    // Create a plan for the backward FFT
    fftw_plan backward_plan = fftw_plan_dft_c2r_1d(n, out, in, FFTW_ESTIMATE);

    // Execute the backward FFT
    fftw_execute(backward_plan);

    // Normalize the result by the size of the signal
    for (int i = 0; i < n; ++i) {
        in[i] /= n;
    }


    // Clean up
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);

    auto result = std::vector(in, in + n);

    fftw_free(in);
    fftw_free(out);

    return result;
}

void fftshift(double* array, int width, int height) {
    int half_width = width / 2;
    int half_height = height / 2;
    int stride = half_width;

    double temp;

    // Shift the top-left quadrant to the center
    for (int y = 0; y < half_height; ++y) {
        for (int x = 0; x < half_width; ++x) {
            int from_index = y * width + x;
            int to_index = (y + half_height) * width + (x + half_width);
            temp = array[from_index];
            array[from_index] = array[to_index];
            array[to_index] = temp;
        }
    }

    // Shift the bottom-left quadrant to the center
    for (int y = half_height; y < height; ++y) {
        for (int x = 0; x < half_width; ++x) {
            int from_index = y * width + x;
            int to_index = (y - half_height) * width + (x + half_width);
            temp = array[from_index];
            array[from_index] = array[to_index];
            array[to_index] = temp;
        }
    }
}

void fftshift(double* array, int nx, int ny, int nz) {

    std::vector<double> src(array, array + nx * ny * nz);

    int shiftX = nx / 2;
    int shiftY = ny / 2;
    int shiftZ = nz / 2;

    for (int i = 0; i < nx; ++i) {
        int ii = (i + shiftX) % nx;
        for (int j = 0; j < ny; ++j) {
            int jj = (j + shiftY) % ny;
            for (int k = 0; k < nz; ++k) {
                int kk = (k + shiftZ) % nz;
                int originalIndex = k + ny * (j + nx * i);
                int shiftedIndex = kk + ny * (jj + nx * ii);
                array[shiftedIndex] = src[originalIndex];
            }
        }
    }
}

double apply_2d_hamming_window(double* signal, int width, int height) {
    double norm = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double hamming_x = 0.54 - 0.46 * cos(2.0 * M_PI * j / (width - 1));
            double hamming_y = 0.54 - 0.46 * cos(2.0 * M_PI * i / (height - 1));
            norm += hamming_x * hamming_y;
            signal[i * width + j] *= hamming_x * hamming_y;
        }
    }
    return norm;
}

template<typename ElemType>
RegularGrid<ElemType> eval_grid(std::function<ElemType(Vec3d)> f, Box3d range, size_t res, size_t num_samples = 1) {
    RegularGrid<ElemType> result = {
        range, res, std::vector<ElemType>(res * res * res)
    };

    for (int i = 0; i < res; i++) {
#pragma omp parallel for
        for (int j = 0; j < res; j++) {
            for (int k = 0; k < res; k++) {
                Vec3d cellMin = lerp(range.min(), range.max(), (Vec3d((double)i, (double)j, (double)k) / res));
                Vec3d cellMax = lerp(range.min(), range.max(), (Vec3d((double)i + 1, (double)j + 1, (double)k + 1) / res));

                UniformPathSampler sampler(0);
                sampler.next2D();

                int gidx = (i * res + j) * res + k;

                if (num_samples > 1) {
                    for (int s = 0; s < num_samples; s++) {
                        result.values[gidx] += f(lerp(cellMin, cellMax, Vec3d(sampler.next1D(), sampler.next1D(), sampler.next1D())));
                    }
                    result.values[gidx] /= num_samples;
                }
                else {
                    result.values[gidx] = f(lerp(cellMin, cellMax, Vec3d(0.5)));
                }
            }
        }
    }

    return result;
}

std::vector<double> compute_acf_fftw_2D(const double* signal, const Vec3d* ps, size_t wx, size_t wy, size_t ww, size_t wh, size_t w, size_t h, std::function<double(Vec3d)> mean) {
    // Allocate memory for the FFTW input and output arrays
    double* in = (double*)fftw_malloc(sizeof(double) * ww * wh);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ww * (wh / 2 + 1));

    // Create a plan for the forward FFT
    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(ww, wh, in, out, FFTW_ESTIMATE);
    // Create a plan for the backward FFT
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(ww, wh, out, in, FFTW_ESTIMATE);

    // Initialize the input array with the signal
    double meanV = 0;
    int idx = 0;
    for (int x = wx; x < wx + ww; x++) {
        for (int y = wy; y < wy + wh; y++) {
            in[idx] = signal[x * w + y] - mean(ps[x * w  + y]);
            meanV += in[idx];
            idx++;
        }
    }

    meanV /= ww * wh;
    for (int i = 0; i < ww * wh; ++i) {
        in[i] -= meanV;
    }

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the power spectrum by squaring the magnitude of the FFT
    for (int i = 0; i < ww * (wh / 2 + 1); ++i) {
        double magnitude = out[i][0] * out[i][0] + out[i][1] * out[i][1];
        out[i][0] = magnitude / (ww * wh);
        out[i][1] = 0.0;
    }

    // Execute the backward FFT
    fftw_execute(backward_plan);

    // Normalize the result by the size of the signal
    for (int i = 0; i < ww * wh; ++i) {
        in[i] /= ww * wh;
    }

    // Clean up
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    
    fftshift(in, ww, wh);
    auto result = std::vector(in, in + ww * wh);

    fftw_free(in);
    fftw_free(out);

    return result;
}

std::vector<double> compute_acf_fftw_3D(
    const double* signal, const Vec3d* ps, 
    size_t wx, size_t wy, size_t wz,
    size_t ww, size_t wh, size_t wd,
    size_t w, size_t h, size_t d,
    std::function<double(Vec3d)> mean) {
    // Allocate memory for the FFTW input and output arrays
    double* in = (double*)fftw_malloc(sizeof(double) * ww * wh * wd);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ww * wh * (wd / 2 + 1));
    fftw_plan forward_plan, backward_plan;
    // Create a plan for the forward FFT
    forward_plan = fftw_plan_dft_r2c_3d(ww, wh, wd, in, out, FFTW_ESTIMATE);
    // Create a plan for the backward FFT
    backward_plan = fftw_plan_dft_c2r_3d(ww, wh, wd, out, in, FFTW_ESTIMATE);


    // Initialize the input array with the signal
    double meanV = 0;
    int idx = 0;
    for (int x = wx; x < wx + ww; x++) {
        for (int y = wy; y < wy + wh; y++) {
            for (int z = wz; z < wz + wd; z++) {
                in[idx] = signal[(x * w + y) * h + z] - mean(ps[(x * w + y) * h + z]);
                meanV += in[idx];
                idx++;
            }
        }
    }

    meanV /= ww * wh * wd;
    for (int i = 0; i < ww * wh * wd; ++i) {
        in[i] -= meanV;
    }

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the power spectrum by squaring the magnitude of the FFT
    for (int i = 0; i < ww * wh * (wd / 2 + 1); ++i) {
        double magnitude = out[i][0] * out[i][0] + out[i][1] * out[i][1];
        out[i][0] = magnitude / (ww * wh * wd);
        out[i][1] = 0.0;
    }

    // Execute the backward FFT
    fftw_execute(backward_plan);

    // Normalize the result by the size of the signal
    for (int i = 0; i < ww * wh * wd; ++i) {
        in[i] /= ww * wh * wd;
    }

    // Clean up
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);

    fftshift(in, ww, wh, wd);
    auto result = std::vector(in, in + ww * wh * wd);

    fftw_free(in);
    fftw_free(out);

    return result;
}

struct CovResidual {
    CovResidual(Vec3d x, double y, std::shared_ptr<StationaryCovariance> cov) :x_(x), y_(y), cov_(cov) {}

    bool operator()(
        const double* const sigma, 
        const double* const ls_x, const double* const ls_y, const double* const ls_z,
        const double* const dir_x, const double* const dir_y, const double* const dir_z,
        double* residual) const {

        ProceduralNonstationaryCovariance mod_cov(
            cov_,
            std::make_shared<ConstantScalar>(sigma[0]),
            std::make_shared<ConstantVector>(Vec3d(ls_x[0], ls_y[0], ls_z[0])),
            std::make_shared<ConstantVector>(Vec3d(dir_x[0], dir_y[0], dir_z[0]).normalized())
        );
        
        residual[0] = y_ - mod_cov(
            Derivative::None, Derivative::None, 
            Vec3d(), x_,
            Vec3d(), Vec3d());

        if (!std::isfinite(residual[0])) {
            residual[0] = 10000;
        }

        return true;
    }
private:
    std::shared_ptr<StationaryCovariance> cov_;
    const Vec3d x_;
    const double y_;
};

// acf is ww x wh array corresponding to the window
// ps is wxh array
// wc is center of the window to use for covariance calculations
std::tuple<ProceduralNonstationaryCovariance, double, Vec3d, Vec3d> fit_cov(const Vec3d* ps, const double* acf, Vec3d wc, size_t wx, size_t wy, size_t ww, size_t wh, size_t w, size_t h, std::shared_ptr<StationaryCovariance> cov) {
    const double initial_sigma = 1.0;
    const Vec3d initial_ls = Vec3d(1.0);
    const Vec3d initial_dir = Vec3d(1.0).normalized();
    
    double sigma = initial_sigma;
    Vec3d ls = initial_ls;
    Vec3d dir = initial_dir;

    ceres::Problem problem;

    for (int lx = ww/2 - ww/8; lx < ww/2 + ww/8; lx++) {
        for (int ly = wh/2 - wh/8; ly < wh/2 + wh/8; ly++) {

            int lidx = lx * ww + ly;
            int gidx = (wx + lx) * w + (wy + ly);

            problem.AddResidualBlock(
                new ceres::NumericDiffCostFunction<CovResidual, ceres::CENTRAL, 1, 1, 1, 1, 1, 1, 1, 1>(
                    new CovResidual(ps[gidx] - wc, acf[lidx], cov)),
                nullptr,
                &sigma,
                &ls.x(), &ls.y(), &ls.z(),
                &dir.x(), &dir.y(), &dir.z());
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial s: " << initial_sigma << " ls: " << initial_ls << " dir: " << initial_dir << "\n";
    std::cout << "Final   s: " << sigma << " ls: " << ls << " dir: " << dir.normalized() << "\n";

    return { ProceduralNonstationaryCovariance(
        cov,
        std::make_shared<ConstantScalar>(sigma),
        std::make_shared<ConstantVector>(ls),
        std::make_shared<ConstantVector>(dir.normalized())
    ), sigma, ls, dir.normalized() };
}


// acf is ww x wh array corresponding to the window
// ps is wxh array
// wc is center of the window to use for covariance calculations
std::tuple<ProceduralNonstationaryCovariance, double, Vec3d, Vec3d> fit_cov_3D(const Vec3d* ps, const double* acf, Vec3d wc, 
    size_t wx, size_t wy, size_t wz, 
    size_t ww, size_t wh, size_t wd, 
    size_t w, size_t h, size_t d,
    std::shared_ptr<StationaryCovariance> cov) {
    const double initial_sigma = 1.0;
    const Vec3d initial_ls = Vec3d(1.0);
    const Vec3d initial_dir = Vec3d(1.0).normalized();

    double sigma = initial_sigma;
    Vec3d ls = initial_ls;
    Vec3d dir = initial_dir;

    ceres::Problem problem;

    for (int lx = ww / 2 - ww / 8; lx < ww / 2 + ww / 8; lx++) {
        for (int ly = wh / 2 - wh / 8; ly < wh / 2 + wh / 8; ly++) {
            for (int lz = wd / 2 - wd / 8; lz < wd / 2 + wd / 8; lz++) {

                int lidx = (lx * ww + ly) * wh + lz;
                int gidx = ((wx + lx) * w + (wy + ly)) * h + (wz + lz);

                problem.AddResidualBlock(
                    new ceres::NumericDiffCostFunction<CovResidual, ceres::CENTRAL, 1, 1, 1, 1, 1, 1, 1, 1>(
                        new CovResidual(ps[gidx] - wc, acf[lidx], cov)),
                    nullptr,
                    &sigma,
                    &ls.x(), &ls.y(), &ls.z(),
                    &dir.x(), &dir.y(), &dir.z());
            }
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return { ProceduralNonstationaryCovariance(
        cov,
        std::make_shared<ConstantScalar>(sigma),
        std::make_shared<ConstantVector>(ls),
        std::make_shared<ConstantVector>(dir.normalized())
    ), sigma, ls, dir.normalized() };
}


std::tuple<Eigen::VectorXd, std::vector<Vec3d>> gen_weight_space_nonstationary(
    std::shared_ptr<GaussianProcess> gp,
    Box3d range,
    size_t res,
    size_t subres, int seed = 0) {

    std::vector<Vec3d> points(res * res);
    {
        int idx = 0;
        for (int i = 0; i < res; i++) {
            for (int j = 0; j < res; j++) {
                points[idx] = lerp(range.min(), range.max(), (Vec3d((double)i, (double)j, 0.) / (res - 1)));
                points[idx][2] = 0.f;
                idx++;
            }
        }
    }

    std::vector<WeightSpaceRealization> realizations(subres * subres);
    {
        UniformPathSampler sampler(seed);
        sampler.next2D();

        int idx = 0;
        for (int i = 0; i < subres; i++) {
            for (int j = 0; j < subres; j++) {
                Vec3d cellCenter = lerp(range.min(), range.max(), (Vec3d((double)i + 0.5, (double)j + 0.5, 0.) / subres));
                cellCenter[2] = 0;

                {
                    auto basis = WeightSpaceBasis::sample(gp->_cov, 300, sampler, cellCenter);
                    realizations[idx] = basis.sampleRealization(gp, sampler);
                }

                idx++;
            }
        }
    }


    auto getValue = [&](Vec2i coord, const Vec3d& p) {
        coord = clamp(coord, Vec2i(0), Vec2i(subres-1));
        return realizations[coord.x() * subres + coord.y()].evaluate(p);
    };

    auto getValues = [&](const Vec2i& coord, const Vec3d& p, double(&data)[2][2]) {
        data[0][0] = getValue(coord + Vec2i(0, 0), p);
        data[1][0] = getValue(coord + Vec2i(1, 0), p);
        data[0][1] = getValue(coord + Vec2i(0, 1), p);
        data[1][1] = getValue(coord + Vec2i(1, 1), p);
    };


    Eigen::VectorXd result(res * res);
    {
        for (int i = 0; i < res; i++) {
            std::cout << i << "\r";
#pragma omp parallel for
            for (int j = 0; j < res; j++) {
                double subres_i = double(i) / (res / subres) - 0.5;
                double subres_j = double(j) / (res / subres) - 0.5;

                double data[2][2];
                getValues(Vec2i((int)floor(subres_i), (int)floor(subres_j)), points[i * res + j], data);

                result[i * res + j] = bilinearInterpolation(Vec2d(subres_i - floor(subres_i), subres_j - floor(subres_j)), data);

            }
        }
    }

    return { result, points };
}


std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> compute_acf_nonstationary(
    std::shared_ptr<GaussianProcess> gp,
    std::function<double(Vec3d)> signal,
    Box3d bounds,
    size_t res,
    size_t subres) {


    std::shared_ptr<StationaryCovariance> base_cov = std::make_shared<SquaredExponentialCovariance>();

    std::vector<double> acf_fit(res * res);
    std::vector<double> acf_gt(res * res);
    std::vector<double> acf_fftw(res * res);
    std::vector<double> residual(res * res);
    std::vector<double> mean(res * res);

    RegularGrid<double> signalGrid = eval_grid(signal, bounds, res);

    auto meanBounds = bounds;
    //meanBounds.grow(bounds.diagonal().length() / (subres * 2 + 1));

    RegularGrid<double> meanGrid = eval_grid<double>([&](Vec3d p) {
        return signalGrid.getValue(p);
    }, meanBounds, subres*2+1);

    auto mean_f = [&](Vec3d p) {
        //return meanGrid.getValue(p);
        
        Derivative d = Derivative::None;
        return gp->mean(&p, &d, nullptr, Vec3d(), 1)(0);
    };

    auto points = signalGrid.makePoints();

    {
        for (int i = 0; i < subres; i++) {
            for (int j = 0; j < subres; j++) {
                Vec3d cellCenter = lerp(bounds.min(), bounds.max(), (Vec3d((double)i + 0.5, (double)j + 0.5, 0.) / subres));
                cellCenter[2] = 0;


                int ww = (res / subres);
                int wh = (res / subres);
                int wx = i * ww;
                int wy = j * wh;


                auto acf = compute_acf_fftw_2D(signalGrid.values.data(), points.data(), wx, wy, ww, wh, res, res, mean_f);
                auto [fitc, var, ls, aniso] = fit_cov(points.data(), acf.data(), cellCenter, wx, wy, ww, wh, res, res, base_cov);

                int idx = 0;
                for (int x = wx; x < wx + ww; x++) {
                    for (int y = wy; y < wy + wh; y++) {
                        acf_fit[x * res + y] = fitc(Derivative::None, Derivative::None, Vec3d(), points[x * res + y] - cellCenter, Vec3d(), Vec3d());
                        acf_gt[x * res + y] = (*gp->_cov)(Derivative::None, Derivative::None, cellCenter, points[x * res + y], Vec3d(), Vec3d());
                        acf_fftw[x * res + y] = acf[idx];
                        residual[x * res + y] = signal(points[x * res + y]) - mean_f(points[x * res + y]);
                        mean[x * res + y] = mean_f(points[x * res + y]);
                        idx++;
                    }
                }
            }
        }
    }

    return { acf_fit, acf_gt, acf_fftw, residual, mean };
}



int estimate_acf(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif
    
    auto cov = std::make_shared<ProceduralNonstationaryCovariance>(
        std::make_shared<SquaredExponentialCovariance>(1.0f, 1.f),
        std::make_shared<ConstantScalar>(1.0),
        std::make_shared<ConstantVector>(Vec3d(1., 0.25, 1.0)),
        std::make_shared<LinearRampVector>(Vec3d(-1.,-1., 0.), Vec3d(1., 1., 0.), Vec3d(1., 0., 0.), Vec2d(-25., 25.))
    );

    //auto cov = std::make_shared<SquaredExponentialCovariance>(1.0, 0.5f);
        

    //auto mean = std::make_shared<SphericalMean>(Vec3d(0.), 20.);
    //auto mean = std::make_shared<LinearMean>(Vec3d(0.), Vec3d(0., 1., 0.), 0.1);
    auto mean = std::make_shared<HomogeneousMean>();

    Path basePath = Path("testing/est-acf/2D");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    size_t dim = 128;
    double scale = 50;

    Box3d range(
        Vec3d(-scale * 0.5), Vec3d(+scale * 0.5)
    );

    auto gp = std::make_shared<GaussianProcess>(mean, cov);
   
    auto [sample, points] = gen_weight_space_nonstationary(gp, range, dim, 16);

    {
        std::ofstream xfile(basePath.asString() + "/ws-real.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
        xfile.close();
    }

    RegularGrid<double> signal{
        range,
        dim,
        std::vector(sample.data(), sample.data() + dim*dim)
    };

    auto signal_f = [&](Vec3d p) {
        return signal.getValue(p);
    };

    auto [acf_fit, acf_gt, acf_fftw, residual, downsampled_mean] = compute_acf_nonstationary(gp, signal_f, range, dim, 4);
    {
        std::ofstream xfile(basePath.asString() + "/acf-fit.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_fit.data(), sizeof(acf_fit[0]) * acf_fit.size());
        xfile.close();
    }

    {
        std::ofstream xfile(basePath.asString() + "/acf-gt.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_gt.data(), sizeof(acf_gt[0]) * acf_gt.size());
        xfile.close();
    }

    {
        std::ofstream xfile(basePath.asString() + "/acf-fftw.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_fftw.data(), sizeof(acf_fftw[0]) * acf_fftw.size());
        xfile.close();
    }

    {
        std::ofstream xfile(basePath.asString() + "/downsampled-mean.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)downsampled_mean.data(), sizeof(downsampled_mean[0]) * downsampled_mean.size());
        xfile.close();
    }

    {
        std::ofstream xfile(basePath.asString() + "/residual.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)residual.data(), sizeof(residual[0]) * residual.size());
        xfile.close();
    }

    return 0;
}


std::tuple<
    std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>,
    RegularGrid<double>, RegularGrid<double>, RegularGrid<Vec3d>, RegularGrid<Vec3d>> compute_acf_nonstationary_real(
    std::function<double(Vec3d)> signal,
    Box3d bounds,
    size_t res,
    size_t meanRes,
    size_t covRes) {


    std::shared_ptr<StationaryCovariance> base_cov = std::make_shared<SquaredExponentialCovariance>();

    std::vector<double> acf_fit(res * res);
    std::vector<double> acf_fftw(res * res);
    std::vector<double> residual(res * res);
    std::vector<double> mean(res * res);

    RegularGrid signalGrid = eval_grid(signal, bounds, res);

    auto cellSize = bounds.diagonal() / meanRes;
    auto meanBounds = bounds;

    RegularGrid meanGrid = eval_grid<double>([&](Vec3d p) {
        return signalGrid.getValue(p);
    }, meanBounds, meanRes, 1000);

    auto mean_f = [&](Vec3d p) {
        return meanGrid.getValue(p);
    };

    auto points = signalGrid.makePoints();

    RegularGrid<double> varGrid{
        bounds, covRes, std::vector<double>(covRes * covRes)
    };

    RegularGrid<Vec3d> lsGrid{
        bounds, covRes, std::vector<Vec3d>(covRes * covRes)
    };

    RegularGrid<Vec3d> anisoGrid{
        bounds, covRes, std::vector<Vec3d>(covRes * covRes)
    };


    {
        for (int i = 1; i < covRes-1; i++) {
            std::cout << i << "\r";
            for (int j = 1; j < covRes-1; j++) {
                Vec3d cellCenter = lerp(bounds.min(), bounds.max(), (Vec3d((double)i + 0.5, (double)j + 0.5, 0.) / covRes));
                cellCenter[2] = 0;

                int ww = (res / covRes)*2;
                int wh = (res / covRes)*2;
                int wx = (i) * (res / covRes) - (res / covRes) / 2;
                int wy = (j) * (res / covRes) - (res / covRes) / 2;


                auto acf = compute_acf_fftw_2D(signalGrid.values.data(), points.data(), wx, wy, ww, wh, res, res, mean_f);
                auto [fitc, var, ls, aniso] = fit_cov(points.data(), acf.data(), cellCenter, wx, wy, ww, wh, res, res, base_cov);

                varGrid.values[i * covRes + j] = var;
                lsGrid.values[i * covRes + j] = ls;
                anisoGrid.values[i * covRes + j] = aniso;

                for (int lx = (res / covRes)/2; lx < 3*(res / covRes)/2; lx++) {
                    for (int ly = (res / covRes)/2; ly < 3*(res / covRes)/2; ly++) {
                        int x = wx + lx;
                        int y = wy + ly;

                        acf_fit[x * res + y] = fitc(Derivative::None, Derivative::None, Vec3d(), points[x * res + y] - cellCenter, Vec3d(), Vec3d());
                        acf_fftw[x * res + y] = acf[lx * ww + ly];
                        residual[x * res + y] = signal(points[x * res + y]) - mean_f(points[x * res + y]);
                        mean[x * res + y] = mean_f(points[x * res + y]);
                    }
                }
            }
        }
    }

    return { acf_fit, acf_fftw, residual, mean, meanGrid, varGrid, lsGrid, anisoGrid };
}


std::tuple<
    RegularGrid<double>, RegularGrid<double>, RegularGrid<double>, RegularGrid<double>,
    RegularGrid<double>, RegularGrid<Vec3d>, RegularGrid<double>, RegularGrid<Vec3d>, RegularGrid<Vec3d>> compute_acf_nonstationary_real_3D(
        std::function<double(Vec3d)> signal,
        std::function<Vec3d(Vec3d)> color,
        Box3d bounds,
        size_t res,
        size_t meanRes,
        size_t covRes) {


    std::shared_ptr<StationaryCovariance> base_cov = std::make_shared<SquaredExponentialCovariance>();

    RegularGrid<double> acf_fit{ bounds, res, std::vector<double>(res * res * res), InterpolateMethod::Point };
    RegularGrid<double> acf_fftw{ bounds, res, std::vector<double>(res * res * res), InterpolateMethod::Point };
    RegularGrid<double> residual{ bounds, res, std::vector<double>(res * res * res), InterpolateMethod::Point };
    RegularGrid<double> mean{ bounds, res, std::vector<double>(res * res * res), InterpolateMethod::Point };

    std::cout << "Evaluating signal on grid\n";
    RegularGrid signalGrid = eval_grid(signal, bounds, res);

    auto meanBounds = bounds;

    std::cout << "Downsampling mean\n";
    RegularGrid meanGrid = eval_grid<double>([&](Vec3d p) {
        return signalGrid.getValue(p);
    }, meanBounds, meanRes, 1000);
    meanGrid.interp = InterpolateMethod::Quadratic;

    std::cout << "Downsampling color\n";
    RegularGrid colorGrid = eval_grid<Vec3d>(color, meanBounds, meanRes, 1000);
    colorGrid.interp = InterpolateMethod::Quadratic;

    auto mean_f = [&](Vec3d p) {
        return meanGrid.getValue(p);
    };

    auto points = signalGrid.makePoints();

    RegularGrid<double> varGrid{
        bounds, covRes, std::vector<double>(covRes * covRes * covRes), InterpolateMethod::Linear
    };

    RegularGrid<Vec3d> lsGrid{
        bounds, covRes, std::vector<Vec3d>(covRes * covRes * covRes), InterpolateMethod::Linear
    };

    RegularGrid<Vec3d> anisoGrid{
        bounds, covRes, std::vector<Vec3d>(covRes * covRes * covRes), InterpolateMethod::Linear
    };

    std::cout << "Compute covariance blocks\n";
    {
        for (int i = 1; i < covRes - 1; i++) {
            std::cout << i << "\r";
#pragma omp parallel for
            for (int j = 1; j < covRes - 1; j++) {
                for (int k = 1; k < covRes - 1; k++) {
                    Vec3d cellCenter = lerp(bounds.min(), bounds.max(), (Vec3d((double)i + 0.5, (double)j + 0.5, (double)k + 0.5) / covRes));

                    int ww = (res / covRes) * 2;
                    int wh = (res / covRes) * 2;
                    int wd = (res / covRes) * 2;

                    int wx = (i) * (res / covRes) - (res / covRes) / 2;
                    int wy = (j) * (res / covRes) - (res / covRes) / 2;
                    int wz = (k) * (res / covRes) - (res / covRes) / 2;

                    std::vector<double> acf;
                    {
                        #pragma omp critical
                        acf = compute_acf_fftw_3D(signalGrid.values.data(), points.data(), wx, wy, wz, ww, wh, wd, res, res, res, mean_f);
                    }

                    auto [fitc, var, ls, aniso] = fit_cov_3D(points.data(), acf.data(), cellCenter, wx, wy, wz, ww, wh, wd, res, res, res, base_cov);

                    varGrid.values[i * covRes * covRes + j * covRes + k] = var;
                    lsGrid.values[i * covRes * covRes + j * covRes + k] = ls;
                    anisoGrid.values[i * covRes * covRes + j * covRes + k] = aniso;

                    for (int lx = (res / covRes) / 2; lx < 3 * (res / covRes) / 2; lx++) {
                        for (int ly = (res / covRes) / 2; ly < 3 * (res / covRes) / 2; ly++) {
                            for (int lz = (res / covRes) / 2; lz < 3 * (res / covRes) / 2; lz++) {
                                int x = wx + lx;
                                int y = wy + ly;
                                int z = wz + lz;

                                int gidx = (x * res + y) * res + z;
                                int lidx = (lx * ww + ly) * wh + lz;

                                acf_fit.values[gidx] = fitc(Derivative::None, Derivative::None, Vec3d(), points[gidx] - cellCenter, Vec3d(), Vec3d());
                                acf_fftw.values[gidx] = acf[lidx];
                                residual.values[gidx] = signal(points[gidx]) - mean_f(points[gidx]);
                                mean.values[gidx] = mean_f(points[gidx]);
                            }
                        }
                    }
                }
            }
        }
    }

    return { acf_fit, acf_fftw, residual, mean, meanGrid, colorGrid, varGrid, lsGrid, anisoGrid };
}


int estimate_acf_mesh(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    Path basePath = Path("testing/est-acf/3d/mesh") / scenePath.stripExtension().baseName();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    size_t dim = 512;

    RegularGrid<double> signal;
    RegularGrid<Vec3d> colors;

    auto sdfPath = basePath / Path(tinyformat::format("sdf-%d.json", dim));
    if (sdfPath.exists()) {
        std::cout << "Loading precomputed SDF\n";
        signal = load_grid<double>(sdfPath);
        signal.interp = InterpolateMethod::Quadratic;

        colors = load_grid<Vec3d>(basePath / Path(tinyformat::format("color-%d.json", dim)));
        colors.interp = InterpolateMethod::Quadratic;

        bool fixedAny = false;
        for (auto& v : colors.values) {
            if (std::isinf(v) || std::isnan(v)) {
                v = Vec3d(0.);
                fixedAny = true;
            }
        }

        if (fixedAny) {
            save_grid(colors, basePath / Path(tinyformat::format("color-%d.json", dim)));
        }
    }
    else {
        std::cout << "Loading mesh\n";
        MeshSdfMean mean(std::make_shared<Path>(argv[1]), true);
        mean.loadResources();

        auto bounds = mean.bounds();
        bounds.grow(bounds.diagonal().length() * 0.2);

        std::cout << "Evaluating high-res mean\n";
        signal = eval_grid<double>([&](Vec3d p) {
            return mean(Derivative::None, p, Vec3d()) - 0.01;
        }, bounds, dim);
        signal.interp = InterpolateMethod::Quadratic;
        save_grid(signal, sdfPath);

        std::cout << "Evaluating high-res color\n";
        colors = eval_grid<Vec3d>([&](Vec3d p) {
            return mean.color(p);
        }, bounds, dim);
        colors.interp = InterpolateMethod::Quadratic;
        save_grid(colors, basePath / Path(tinyformat::format("color-%d.json", dim)));
    }

    int mean_res = 64;
    int cov_res = 16;

    auto signal_f = [&](Vec3d p) {
        return signal.getValue(p);
    };

    auto color_f = [&](Vec3d p) {
        return colors.getValue(p);
    };

    auto occupancy_highres = eval_grid<double>([&](Vec3d p) {
        return signal_f(p) < 0 ? 1. : 0.;
    }, signal.bounds, dim, 1);
    occupancy_highres.interp = InterpolateMethod::Quadratic;
    save_grid(occupancy_highres, basePath / "occupancy-high.json");

    auto occupancy_avg = eval_grid<double>([&](Vec3d p) {
        return signal_f(p) < 0 ? 1. : 0.;
    }, signal.bounds, mean_res, 1000);
    occupancy_avg.interp = InterpolateMethod::Quadratic;
    save_grid(occupancy_avg, basePath / "occupancy-avg.json");


    std::cout << "Computing ACF\n";
    auto [acf_fit, acf_fftw, residual, downsampled_mean, 
        mean, color, variance, ls, aniso] = compute_acf_nonstationary_real_3D(signal_f, color_f, signal.bounds, dim, mean_res, cov_res);

    save_grid(mean, basePath / "mean.json");
    save_grid(color, basePath / "color.json");
    save_grid(variance, basePath / "var.json");
    save_grid(ls, basePath / "ls.json");
    save_grid(aniso, basePath / "aniso.json");

    save_grid(acf_fit, basePath / "acf-fit.json");
    save_grid(acf_fftw, basePath / "acf-fftw.json");
    save_grid(downsampled_mean, basePath / "downsampled-mean.json");
    save_grid(residual, basePath / "residual.json");

    return 0;
}

int inspect_acf_gp(int argc, char** argv) {
    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    Path basePath = Path("testing/est-acf/3d/mesh") / scenePath.stripExtension().baseName();
    if (!basePath.exists()) {
        return 0;
    }

    size_t dim = 512;

    auto mean = load_grid<double>(basePath / "mean.json");
    auto var = load_grid<double>(basePath / "var.json");
    auto ls = load_grid<Vec3d>(basePath / "ls.json");
    auto aniso = load_grid<Vec3d>(basePath / "aniso.json");

    auto gp = std::make_shared<GaussianProcess>(
        std::make_shared<ProceduralMean>(std::make_shared<RegularGridScalar>(std::make_shared<RegularGrid<double>>(mean))),
        std::make_shared<ProceduralNonstationaryCovariance>(
            std::make_shared<SquaredExponentialCovariance>(),
            std::make_shared<RegularGridScalar>(std::make_shared<RegularGrid<double>>(var)),
            std::make_shared<RegularGridVector>(std::make_shared<RegularGrid<Vec3d>>(ls)),
            std::make_shared<RegularGridVector>(std::make_shared<RegularGrid<Vec3d>>(aniso))
        )
    );

    /*auto gp = std::make_shared<GaussianProcess>(
        std::make_shared<ProceduralMean>(std::make_shared<RegularGridScalar>(std::make_shared<RegularGrid<double>>(mean))),
        std::make_shared<ProceduralNonstationaryCovariance>(
            std::make_shared<SquaredExponentialCovariance>(),
            std::make_shared<RegularGridScalar>(std::make_shared<RegularGrid<double>>(var)),
            std::make_shared<ConstantVector>(Vec3d(0.5)),
            nullptr
        )
    );*/

    auto occupancyGrid = eval_grid<double>([&](Vec3d p) {
        return gp->cdf(p);
    }, var.bounds, dim, 1);

    save_grid(occupancyGrid, basePath / "occupancy.json");

    /*for (int i = 0; i < 10; i++) {
        auto [samples, points] = gen_weight_space_nonstationary(gp, var.bounds, dim, 8, i);

        save_grid(
            RegularGrid<double>(var.bounds, dim, std::vector(samples.data(), samples.data() + samples.size())), 
            basePath / tinyformat::format("sample-%d.json", i));
    }*/

    return 0;
}


int main(int argc, char** argv) {
    //return mesh_convert_2d(argc, argv);
    return mesh_convert(argc, argv);
    //return gen3d(argc, argv);
    //return test2d(argc, argv);

    //return estimate_acf(argc, argv);
    //estimate_acf_mesh(argc, argv);
    //return inspect_acf_gp(argc, argv);
}
