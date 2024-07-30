#include "Gaussian.hpp"

#include "math/Angle.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <random>
#include <cmath>

namespace Tungsten {
    CovMatrix project_to_psd(const CovMatrix& in) {

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(in);
        auto eps = 1e6 * std::numeric_limits<double>::epsilon() * eigs.eigenvalues()[0];

        CovMatrix result = eigs.eigenvectors()
            * eigs.eigenvalues().array().max(eps).matrix() * eigs.eigenvectors();
        //result.diagonal().array() += 1e-6;
        return result;
    }

    // Box muller transform
    Vec2d rand_normal_2(PathSampleGenerator& sampler) {
        double u1 = sampler.next1D();
        double u2 = sampler.next1D();

        double r = sqrt(-2 * log(1. - u1));
        double x = cos(2 * PI * u2);
        double y = sin(2 * PI * u2);
        double z1 = r * x;
        double z2 = r * y;

        return Vec2d(z1, z2);

    }

    double rand_gamma(double shape, double mean, PathSampleGenerator& sampler) {
        double scale = mean / shape;
        // Not ideal
        std::mt19937 rnd(sampler.nextDiscrete(1 << 16));
        std::gamma_distribution<> gamma_dist(shape, scale);
        return gamma_dist(rnd);
    }

    double rand_truncated_normal(double mean, double sigma, double a, PathSampleGenerator& sampler) {
        if (abs(a - mean) < 0.000001) {
            return abs(mean + sigma * rand_normal_2(sampler).x());
        }

        if (a < mean) {
            while (true) {
                double x = mean + sigma * rand_normal_2(sampler).x();
                if (x >= a) {
                    return x;
                }
            }
        }

        double a_bar = (a - mean) / sigma;
        double x_bar;

        for(int i = 0; i < 1000; i++) {
            double u = sampler.next1D();
            x_bar = sqrt(a_bar * a_bar - 2 * log(1 - u));
            double v = sampler.next1D();

            if (v < x_bar / a_bar) {
                break;
            }
        }

        return sigma * x_bar + mean;
    }

    Eigen::VectorXd sample_standard_normal(int n, PathSampleGenerator& sampler) {
        Eigen::VectorXd result(n);
        // We're always getting two samples, so make use of that
        for (int i = 0; i < result.size() / 2; i++) {
            Vec2d norm_samp = rand_normal_2(sampler);
            result(i * 2) = norm_samp.x();
            result(i * 2 + 1) = norm_samp.y();
        }

        // Fill up the last one for an uneven number of samples
        if (result.size() % 2) {
            Vec2d norm_samp = rand_normal_2(sampler);
            result(result.size() - 1) = norm_samp.x();
        }
        return result;
    }

    MultivariateNormalDistribution::MultivariateNormalDistribution(const Eigen::VectorXd& _mean, const CovMatrix& _cov) : mean(_mean) {
#if 0
        svd = Eigen::BDCSVD<Eigen::MatrixXd>(_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

        if (svd.info() != Eigen::Success) {
            std::cerr << "SVD for MVN computations failed!\n";
        }

        double logDetCov = 0;
        for (int i = 0; i < svd.nonzeroSingularValues(); i++) {
            logDetCov += log(svd.singularValues()(i));
        }
        sqrt2PiN = std::exp(logDetCov);

        // Compute the square root of the PSD matrix
        normTransform = svd.matrixU() * svd.singularValues().array().max(0).sqrt().matrix().asDiagonal() * svd.matrixV().transpose();

#else

#ifdef SPARSE_COV
        Eigen::SimplicialLLT<CovMatrix> chol(_cov);
#else
        Eigen::LLT<Eigen::MatrixXd> chol(_cov.triangularView<Eigen::Lower>());
#endif

        // We can only use the cholesky decomposition if 
        // the covariance matrix is symmetric, pos-definite.
        // But a covariance matrix might be pos-semi-definite.
        // In that case, we'll go to an EigenSolver
        if (chol.info() == Eigen::Success) {
            // Use cholesky solver
            normTransform = chol.matrixL();
        }
        else
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(_cov);

            if (eigs.info() != Eigen::ComputationInfo::Success) {
                std::cerr << "Matrix square root failed!\n";
            }

            normTransform = eigs.eigenvectors()
                * eigs.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
        }
#endif

    }

    double MultivariateNormalDistribution::eval(const Eigen::VectorXd& x) const {
        Eigen::VectorXd diff = x - mean;

        double quadform = diff.transpose() * svd.solve(diff);

        double inv_sqrt_2pi = 0.3989422804014327;
        double normConst = pow(inv_sqrt_2pi, x.rows()) * pow(sqrt2PiN, -.5);
        return normConst * exp(-.5 * quadform);
    }

    Eigen::MatrixXd MultivariateNormalDistribution::sample(const Constraint* constraints, int numConstraints,
        int samples, PathSampleGenerator& sampler) const {

        // Generate a vector of standard normal variates with the same dimension as the mean
        Eigen::VectorXd z = Eigen::VectorXd(mean.size());
        Eigen::MatrixXd sample(mean.size(), samples);

        int numTries = 0;
        for (int j = 0; j < samples; /*only advance sample idx if the sample passes all constraints*/) {

            numTries++;

            // We're always getting two samples, so make use of that
            for (int i = 0; i < mean.size() / 2; i++) {
                Vec2d norm_samp = rand_normal_2(sampler); // { (float)random_standard_normal(sampler), (float)random_standard_normal(sampler) };
                z(i * 2) = norm_samp.x();
                z(i * 2 + 1) = norm_samp.y();
            }

            // Fill up the last one for an uneven number of samples
            if (mean.size() % 2) {
                Vec2d norm_samp = rand_normal_2(sampler);
                z(mean.size() - 1) = norm_samp.x();
            }

            Eigen::VectorXd currSample = mean + normTransform * z;

            // Check constraints
            bool passedConstraints = true;
            for (int cIdx = 0; cIdx < numConstraints; cIdx++) {
                const Constraint& con = constraints[cIdx];

                for (int i = con.startIdx; i <= con.endIdx; i++) {
                    if (currSample(i) < con.minV || currSample(i) > con.maxV) {
                        passedConstraints = false;
                        break;
                    }
                }

                if (!passedConstraints) {
                    break;
                }
            }

            if (passedConstraints || numTries > 100000) {
                if (numTries > 100000) {
                    std::cout << "Constraint not satisfied. " << mean(0) << "\n";
                }
                sample.col(j) = currSample;
                j++;
                numTries = 0;
            }
        }

        return sample;
    }
}