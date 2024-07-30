#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <ccomplex>
#include <fftw3.h>
#include <tinyformat/tinyformat.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/MultiResGrid.h>

using namespace Tungsten;

constexpr int NUM_SAMPLE_POINTS = 2048;
constexpr int NUM_PT_EVAL_POINTS = 512;

void fft_shift(const std::vector<std::complex<double>>& dft, std::vector<std::complex<double>>& dft_shift, int offset) {
    for (int i = 0; i < dft.size(); i++)
    {
        int src = i;
        int dst = (i + dft.size() / 2 - 1) % dft.size();

        dft_shift[dst+offset] = dft[src];
    }
}

std::vector<std::complex<double>> pad_both_sides(const std::vector<std::complex<double>>& dft, int new_size) {
    int new_elems = new_size - dft.size();
    int pad_elems = new_elems / 2;

    std::vector<std::complex<double>> result(new_size, 0);

    fft_shift(dft, result, pad_elems);

    std::vector<std::complex<double>> result2(new_size);

    fft_shift(result, result2, 0);
    return result2;
}

float compute_freq(int idx, int spc, int res) {
    int shifted_idx = (idx + spc / 2 - 1) % spc;
    int orig_array_idx = shifted_idx + (res - spc) / 2;
    int freq_idx = (orig_array_idx - (res / 2 - 1)) % res;
    return (float)freq_idx / (float)res;
}

float eval_idft_point(Vec3f p, const std::vector<std::complex<double>>& dft) {

    int res = NUM_PT_EVAL_POINTS;
    p = (p * res);
    for (int i = 0; i < 3; i++) {
        p[i] = floor(p[i]);
    }


    float sum = 0;
    int idx = 0;
    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        float u = compute_freq(i, NUM_SAMPLE_POINTS, res);

        for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
            float v = compute_freq(j, NUM_SAMPLE_POINTS, res);

            float s = 2.0f * PI * (u * p.x() + v * p.y());
            sum += dft[idx].real() * std::cos(s) - dft[idx].imag() * std::sin(s);
            idx++;
        }
    }

    return sum;
}



void real_2D(const GaussianProcess& gp) {

    Path basePath = Path("2d-reals") / Path(gp._cov->id());

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    UniformPathSampler sampler(0);
    sampler.next2D();


    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(points.size());

    std::vector<std::complex<double>> cov(points.size());
    std::vector<std::complex<double>> Fcov(points.size());
    fftw_plan plan = fftw_plan_dft_2d(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, (fftw_complex*)cov.data(), (fftw_complex*)Fcov.data(), FFTW_FORWARD, FFTW_ESTIMATE);

    int idx = 0;
    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
            points[idx] = 4. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
            points[idx][2] = 0.f;

            derivs[idx] = Derivative::None;
            cov[idx] = std::complex<double>((*gp._cov)(Derivative::None, Derivative::None, Vec3d(0.f), points[idx], -points[idx].normalized(), points[idx].normalized()), 0.f);
            idx++;
        }
    }

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    std::vector<std::complex<double>> Fcov_sample(points.size());
    std::vector<std::complex<double>> real(points.size());
    plan = fftw_plan_dft_2d(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, (fftw_complex*)Fcov_sample.data(), (fftw_complex*)real.data(), FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int num_reals = 0; num_reals < 1; num_reals++) {
        for (int i = 0; i < Fcov.size(); i++) {
            Vec2d u = rand_normal_2(sampler);
            Fcov_sample[i] = sqrt(Fcov[i] / std::complex<double>(cov.size())) * (std::complex<double>(u.x(), u.y()));
        }

        fftw_execute(plan);

        for (int i = 0; i < Fcov.size(); i++) {
            Vec2d u = rand_normal_2(sampler);
            real[i] += std::complex<double>((*gp._mean)(derivs[i], points[i], Vec3d(0.f)));
        }

        {
            std::ofstream xfile( (basePath / Path(tinyformat::format("sample-%d-%d.bin", NUM_SAMPLE_POINTS, num_reals))).asString(), std::ios::out | std::ios::binary);
            xfile.write((char*)real.data(), sizeof(fftw_complex) * real.size());
            xfile.close();
        }
    }

    for (int i = 0; i < Fcov.size(); i++) {
        Vec2d u = rand_normal_2(sampler);
        real[i] = std::complex<double>((*gp._mean)(derivs[i], points[i], Vec3d(0.f)));
    }

    {
        std::ofstream xfile((basePath / Path(tinyformat::format("mean-%d-%d.bin", NUM_SAMPLE_POINTS, 0))).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)real.data(), sizeof(fftw_complex) * real.size());
        xfile.close();
    }

    fftw_destroy_plan(plan);
}

void rational_quadratic_sphere_3D() {
    GaussianProcess gp(std::make_shared<SphericalMean>(Vec3d(0.0f, 0.0f, 0.f), 1.f), std::make_shared<RationalQuadraticCovariance>(0.5f, 0.2f, 1.0f, Vec3f(1.f, 1.0f, 1.f)));
    UniformPathSampler sampler(0);
    sampler.next2D();


    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(points.size());

    std::vector<std::complex<double>> cov(points.size());
    std::vector<std::complex<double>> Fcov(points.size());
    fftw_plan plan = fftw_plan_dft_3d(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS,
        (fftw_complex*)cov.data(), (fftw_complex*)Fcov.data(), FFTW_FORWARD, FFTW_ESTIMATE);

    int idx = 0;
    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
            for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                points[idx] = 4. * (Vec3d((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS - 1) - 0.5f);

                derivs[idx] = Derivative::None;
                cov[idx] = std::complex<double>((*gp._cov)(Derivative::None, Derivative::None, Vec3d(0.f), points[idx], -points[idx].normalized(), points[idx].normalized()), 0.f);
                idx++;
            }
        }
    }

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    std::vector<std::complex<double>> Fcov_sample(points.size());
    std::vector<std::complex<double>> real(points.size());
    plan = fftw_plan_dft_3d(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 
        (fftw_complex*)Fcov_sample.data(), (fftw_complex*)real.data(), FFTW_BACKWARD, FFTW_MEASURE);

    {
        std::ofstream xfile(tinyformat::format("3d-reals/%s-sample.bin", gp._cov->id()), std::ios::out | std::ios::binary);

        for (int num_reals = 0; num_reals < 5; num_reals++) {
            openvdb::io::File file(tinyformat::format("3d-reals/%s-sample%d.vdb", gp._cov->id(), num_reals));

            for (int i = 0; i < Fcov.size(); i++) {
                Vec2d u = rand_normal_2(sampler);
                Fcov_sample[i] = sqrt(Fcov[i] / std::complex<double>(4*cov.size())) * (std::complex<double>(u.x(), u.y()));
            }

            fftw_execute(plan);

            for (int i = 0; i < Fcov.size(); i++) {
                Vec2d u = rand_normal_2(sampler);
                real[i] += std::complex<double>((*gp._mean)(derivs[i], points[i], Vec3d(0.f)));
            }


            auto grid = openvdb::createGrid<openvdb::FloatGrid>(4.f);
            grid->setGridClass(openvdb::GRID_LEVEL_SET);

            openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

            const float outside = grid->background();
            const float inside = -outside;

            std::cout << outside << "\n";

            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                        float val = real[idx].real();
                        if (abs(val) < outside*10) {
                            accessor.setValue({ i,j,k }, val);
                        }
                        idx++;
                    }
                }
            }

            openvdb::GridPtrVec grids;
            grid->setName("density");
            grids.push_back(grid);
            file.write(grids);
            file.close();

            xfile.write((char*)real.data(), sizeof(fftw_complex) * real.size());
        }

        xfile.close();
    }

    for (int i = 0; i < Fcov.size(); i++) {
        Vec2d u = rand_normal_2(sampler);
        real[i] = std::complex<double>((*gp._mean)(derivs[i], points[i], Vec3d(0.f)));
    }

    {
        std::ofstream xfile(tinyformat::format("3d-reals/%s-mean.bin", gp._cov->id()), std::ios::out | std::ios::binary);
        xfile.write((char*)real.data(), sizeof(fftw_complex) * real.size());
        xfile.close();
    }

    fftw_destroy_plan(plan);
}


int main() {
    try {
        GaussianProcess gp(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.0f, 0.25f, Vec3f(1.f, 1.f, 1.f)));
        real_2D(gp);
    }
    catch (std::exception& e) {
        std::cerr << e.what();
    }
}



#if 0

GaussianProcess gp(std::make_shared<SphericalMean>(Vec3f(0.f, 0.f, 0.f), 0.25f), std::make_shared<RationalQuadraticCovariance>(1.0f, 0.1f, 0.25f, Vec3f(1.f, 1.0f, 1.f)));

UniformPathSampler sampler(0);
sampler.next2D();


std::vector<Vec3f> points(NUM_SAMPLE_POINTS* NUM_SAMPLE_POINTS);
std::vector<Derivative> derivs(points.size());

std::vector<std::complex<double>> cov(points.size());
std::vector<std::complex<double>> Fcov(points.size());
fftw_plan plan = fftw_plan_dft_2d(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, (fftw_complex*)cov.data(), (fftw_complex*)Fcov.data(), FFTW_FORWARD, FFTW_ESTIMATE);

int idx = 0;
for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
    for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
        points[idx] = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
        points[idx][2] = 0.f;

        derivs[idx] = Derivative::None;
        cov[idx] = std::complex<double>((*gp._cov)(Derivative::None, Derivative::None, Vec3f(0.f), points[idx], points[idx].normalized()), 0.f);
        idx++;
    }
}


fftw_execute(plan);

fftw_destroy_plan(plan);

std::vector<std::complex<double>> Fcov_sample(points.size());
std::vector<std::complex<double>> real(points.size());
plan = fftw_plan_dft_2d(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, (fftw_complex*)Fcov_sample.data(), (fftw_complex*)real.data(), FFTW_BACKWARD, FFTW_ESTIMATE);

idx = 0;
for (int i = 0; i < Fcov.size(); i++) {
    Vec2f u = gp.rand_normal_2(sampler);
    Fcov_sample[idx] = sqrt(Fcov[idx] / std::complex<double>(cov.size(), 0.f)) * (std::complex<double>(u.x(), u.y()));
    idx++;
}


std::vector<double> real_pteval(NUM_PT_EVAL_POINTS * NUM_PT_EVAL_POINTS);
idx = 0;
for (int i = 0; i < NUM_PT_EVAL_POINTS; i++) {
    for (int j = 0; j < NUM_PT_EVAL_POINTS; j++) {
        Vec3f p = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_PT_EVAL_POINTS - 1) - 0.5f);
        p[2] = 0.f;

        p = p * 0.5f + 0.5f;

        real_pteval[idx] = eval_idft_point(p, Fcov_sample);
        idx++;
    }
}

fftw_execute(plan);
fftw_destroy_plan(plan);

{
    std::ofstream xfile("fftw-input.bin", std::ios::out | std::ios::binary);
    xfile.write((char*)cov.data(), sizeof(fftw_complex) * cov.size());
    xfile.close();
}

{
    std::ofstream xfile("fftw-result.bin", std::ios::out | std::ios::binary);
    xfile.write((char*)Fcov.data(), sizeof(fftw_complex) * Fcov.size());
    xfile.close();
}

{
    std::ofstream xfile("fftw-transform.bin", std::ios::out | std::ios::binary);
    xfile.write((char*)Fcov_sample.data(), sizeof(fftw_complex) * Fcov_sample.size());
    xfile.close();
}

{
    std::ofstream xfile("fftw-sample.bin", std::ios::out | std::ios::binary);
    xfile.write((char*)real.data(), sizeof(fftw_complex) * real.size());
    xfile.close();
}

{
    std::ofstream xfile("fftw-sample-pteval.bin", std::ios::out | std::ios::binary);
    xfile.write((char*)real_pteval.data(), sizeof(double) * real_pteval.size());
    xfile.close();
}


{
    int idx = 0;
    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
            for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                points[idx] = 2.f * (Vec3f((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                derivs[idx] = Derivative::None;
                idx++;
            }
        }
    }

    Eigen::MatrixXf samples = gp.sample(
        points.data(), derivs.data(), points.size(),
        nullptr, 0,
        Vec3f(1.0f, 0.0f, 0.0f), 1, sampler);

    {
        std::ofstream xfile("grid-samples.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }
}
#endif