#ifndef GAUSSIAN_HPP_
#define GAUSSIAN_HPP_

#include "math/MathUtil.hpp"
#include "sampling/PathSampleGenerator.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace Tungsten {

    //#define SPARSE_COV
#ifdef SPARSE_COV
    using CovMatrix = Eigen::SparseMatrix<double>;
#else
    using CovMatrix = Eigen::MatrixXd;
#endif

    CovMatrix project_to_psd(const CovMatrix& in);

    // Box muller transform
    Vec2d rand_normal_2(PathSampleGenerator& sampler);
    double rand_truncated_normal(double mean, double sigma, double a, PathSampleGenerator& sampler);
    double rand_gamma(double shape, double mean, PathSampleGenerator& samples);

    struct Constraint {
        int startIdx, endIdx;
        float minV, maxV;
    };

    Eigen::VectorXd sample_standard_normal(int n, PathSampleGenerator& sampler);

    struct MultivariateNormalDistribution {
        Eigen::VectorXd mean;

        Eigen::BDCSVD<Eigen::MatrixXd> svd;
        //Eigen::LLT<Eigen::MatrixXd> chol;

        double sqrt2PiN;

        Eigen::MatrixXd normTransform;

        MultivariateNormalDistribution(const Eigen::VectorXd& _mean, const CovMatrix& _cov);

        double eval(const Eigen::VectorXd& x) const;

        Eigen::MatrixXd sample(const Constraint* constraints, int numConstraints,
            int samples, PathSampleGenerator& sampler) const;
    };

}

#endif /* GAUSSIAN_HPP_ */
