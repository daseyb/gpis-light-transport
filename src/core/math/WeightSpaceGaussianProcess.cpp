#include "WeightSpaceGaussianProcess.hpp"

#include <sampling/SampleWarp.hpp>
#include <math/TangentFrame.hpp>

namespace Eigen {
    template<class T>
    void swap(T&& a, T&& b) {
        a.swap(b);
    }
}

namespace Tungsten {

WeightSpaceRealization WeightSpaceRealization::truncate(size_t n) const {
    auto trunc_basis = basis->truncate(n);

    return WeightSpaceRealization{
        std::make_shared<WeightSpaceBasis>(trunc_basis),
        gp,
        weights.block(0, 0, n, 1)
    };
}

double WeightSpaceRealization::evaluate(const Vec3d& p) const {
    Derivative d = Derivative::None;
    double c = (*gp->_cov)(Derivative::None, Derivative::None, p, p, Vec3d(), Vec3d());
    double scale = sqrt(c);
    return scale * basis->evaluate(vec_conv<Eigen::Vector3d>(gp->shell_embedding(p)), weights) + gp->mean_prior(&p, &d, nullptr, Vec3d(0.), 1)(0);
}

Affine<1> WeightSpaceRealization::evaluate(const Affine<3>& p) const {
    Derivative d = Derivative::None;
    
    // Assume constant variance
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, vec_conv<Vec3d>(p.base), vec_conv<Vec3d>(p.base), Vec3d(), Vec3d()));
    auto basisRes = basis->evaluate(p, weights) * scale; 
    auto np = p;
    np.aff.resize(Eigen::NoChange, std::max(basisRes.aff.cols(), p.aff.cols()));
    np.aff.setZero();
    np.aff.block(0, 0, 3, p.aff.cols()) = p.aff; 

    basisRes += gp->_mean->mean(np);
    return basisRes;
}

Vec3d WeightSpaceRealization::evaluateGradient(const Vec3d& p) const {
    auto get_jacobian = [&](auto f, Vec3d p) {
        auto c = f(p);
        double eps = 0.0001;
        Eigen::Matrix3d jac;
        do {
            jac.row(0) = vec_conv<Eigen::Vector3d>((f(p + Vec3d(eps, 0., 0.)) - c) / eps);
            jac.row(1) = vec_conv<Eigen::Vector3d>((f(p + Vec3d(0., eps, 0.)) - c) / eps);
            jac.row(2) = vec_conv<Eigen::Vector3d>((f(p + Vec3d(0., 0., eps)) - c) / eps);
            eps *= 2;
        } while(jac.determinant() < 0.0001);
        return jac;
    };

    Derivative d = Derivative::None;
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, p, p, Vec3d(), Vec3d()));

    Vec3d basisGrad = scale * basis->evaluateGradient(vec_conv<Eigen::Vector3d>(gp->shell_embedding(p)), weights);
    Eigen::Matrix3d jac = get_jacobian([&](Vec3d p) {return gp->shell_embedding(p); }, p);
    basisGrad = vec_conv<Vec3d>(jac.inverse().transpose() * vec_conv<Eigen::Vector3d>(basisGrad));
    
    return basisGrad + gp->_mean->dmean_da(p);
}


Eigen::VectorXd WeightSpaceRealization::evaluate(const Vec3d* ps, size_t num_ps) const {
    Eigen::VectorXd result(num_ps);
    for (size_t p = 0; p < num_ps; p++) {
        result[p] = evaluate(ps[p]);
    }
    return result;
}

double WeightSpaceRealization::lipschitz() const {
    return sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d())) * basis->lipschitz(weights) + gp->_mean->lipschitz();
}

RangeBound WeightSpaceRealization::rangeBound(const Vec3d& c, const std::vector<Vec3d>& vs) const {

    Affine<3> test(c, vs);

    Affine<1> res = evaluate(test);

    auto r = res.mayContainBounds();

    if (r.lower(0) > 0) return RangeBound::Positive;
    else if (r.upper(0) < 0) return RangeBound::Negative;
    else return RangeBound::Unknown;
}

Affine<1> WeightSpaceBasis::evaluate(const Affine<3>& p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    auto np = p;

    Affine<1> result = 0;
    for (size_t row = 0; row < size(); row++) {
        result += aff_cos(dot<3,double>(dirs.row(row), np) * freqs[row] + offsets[row]) * weights[row];
        if (result.mode != Affine<1>::Mode::AffineFixed) {
            np.aff.resize(Eigen::NoChange, result.aff.cols());
            np.aff.setZero();
            np.aff.block(0, 0, 3, p.aff.cols()) = p.aff;
        }
    }
    return result * sqrt(2. / size());
}

double WeightSpaceBasis::evaluate(Eigen::Vector3d p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    double result = 0;
    for (size_t row = 0; row < size(); row++) {
        result += weights[row] * cos(dirs.row(row).dot(p) * freqs[row] + offsets[row]);
    }
    return result * sqrt(2. / size()) ;
}

Vec3d WeightSpaceBasis::evaluateGradient(Eigen::Vector3d p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return Vec3d(0.);

    Vec3d result = Vec3d(0.);
    for (size_t row = 0; row < size(); row++) {
        result += vec_conv<Vec3d>(dirs.row(row)) * freqs[row] * weights[row] * -sin(dirs.row(row).dot(p) * freqs[row] + offsets[row]);
    }
    return result * sqrt(2. / size());
}

Eigen::MatrixXd WeightSpaceBasis::phi(Eigen::MatrixXd ps, const Eigen::VectorXd& weights) const {
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(ps.rows(), size());
    if (size() == 0) return phi;
    for (size_t p = 0; p < ps.rows(); p++) {
        for (size_t row = 0; row < size(); row++) {
            phi(p, row) = cos(dirs.row(row).dot(ps.row(p)) * freqs[row] + offsets[row]);
        }
    }
    return phi * sqrt(2. / size());
}

double WeightSpaceBasis::lipschitz(const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    double lipschitz = 0;
    for (size_t row = 0; row < size(); row++) {
        lipschitz += std::abs(weights[row] * freqs[row]);
    }
    return lipschitz * sqrt(2. / size());
}


WeightSpaceBasis WeightSpaceBasis::sample(std::shared_ptr<CovarianceFunction> cov, int n, PathSampleGenerator& sampler, Vec3d spectralLoc, bool sort, int d) {
    WeightSpaceBasis b(n);
    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame({0., 0., 1.});

    for (int i = 0; i < n; i++) {
        b.offsets(i) = sampler.next1D() * TWO_PI;
        
        if (d == 1) {
            b.freqs(i) = cov->sample_spectral_density(sampler, spectralLoc);
            auto xi = sampler.next2D();
            auto dir = SampleWarp::uniformCylinder(xi);
            dir.z() = 0;
            b.dirs.row(i) = vec_conv<Eigen::Vector3d>(dir);
        }
        else if (d == 2) {
            auto dir2d = cov->sample_spectral_density_2d(sampler, spectralLoc);
            auto dir = Vec3d(dir2d.x(), dir2d.y(), 0.);

            b.dirs.row(i) = frame.toGlobal(vec_conv<Eigen::Vector3d>(dir.normalized()));
            b.freqs(i) = dir.length();

            if (!std::isfinite(b.freqs(i)) || dir.length() < 0.000001) {
                std::cerr << "Sampling error!\n";
                i--;
                continue;
            }
        }
        else {
            auto dir = cov->sample_spectral_density_3d(sampler, spectralLoc);

            b.dirs.row(i) = vec_conv<Eigen::Vector3d>(dir.normalized());
            b.freqs(i) = dir.length();

            if (!std::isfinite(b.freqs(i))) {
                std::cerr << "Sampling error!\n";
            }
        }
    }

    if (sort) {
        Eigen::MatrixXd data(n, b.freqs.cols() + b.offsets.cols() + b.dirs.cols());
        data.block(0, 0, n, 1) = b.freqs;
        data.block(0, 1, n, 1) = b.offsets;
        data.block(0, 2, n, 3) = b.dirs;

        auto data_rows = data.rowwise();
        std::sort(data_rows.begin(), data_rows.end(),
            [](auto const& r1, auto const& r2) {return r1(0) < r2(0); });

        b.freqs = data.block(0, 0, n, b.freqs.cols());
        b.offsets = data.block(0, 1, n, b.offsets.cols());
        b.dirs = data.block(0, 2, n, b.dirs.cols());
    }
    return b;
}

WeightSpaceBasis WeightSpaceBasis::create_regular(std::shared_ptr<CovarianceFunction> cov, int n, float maxFreq, PathSampleGenerator& sampler, Vec3d spectralLoc) {
    WeightSpaceBasis b(n);
    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame({ 0., 0., 1. });

    for (int i = 0; i < n; i++) {
        b.offsets(i) = sampler.next1D() * TWO_PI;

        float s =  maxFreq * (i + 1) / n;

        b.freqs(i) = cov->spectral_density(s);
        auto xi = sampler.next2D();
        b.dirs.row(i) = vec_conv<Eigen::Vector3d>(SampleWarp::uniformSphere(sampler.next2D()));
    }

    return b;
}

WeightSpaceRealization WeightSpaceRealization::sample(std::shared_ptr<WeightSpaceBasis> basis, std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) {
    return WeightSpaceRealization{
        basis, gp, sample_standard_normal(basis->size(), sampler)
    };
}


WeightSpaceRealization WeightSpaceBasis::sampleRealization(std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) const {
    return WeightSpaceRealization{
        std::make_shared<WeightSpaceBasis>(*this), gp, sample_standard_normal(size(), sampler)
    };
}

WeightSpaceBasis WeightSpaceBasis::truncate(size_t n) const {
    return WeightSpaceBasis(
        dirs.block(0, 0, n, 3),
        freqs.block(0, 0, n, 1),
        offsets.block(0, 0, n, 1)
    );
}

}