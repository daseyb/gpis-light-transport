#ifndef GAUSSIANPROCESS_HPP_
#define GAUSSIANPROCESS_HPP_
#include "sampling/PathSampleGenerator.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "BitManip.hpp"
#include "math/Vec.hpp"
#include "math/MathUtil.hpp"
#include "math/Angle.hpp"
#include "sampling/SampleWarp.hpp"
#include <math/GPFunctions.hpp>
#include <sampling/Gaussian.hpp>

#include <functional>
#include <vector>
#include <variant>

#include "io/JsonSerializable.hpp"
#include "io/JsonObject.hpp"

namespace Tungsten {

    class GaussianProcess;

    enum class GPCorrelationContext {
        Elephant,
        Goldfish,
        Dori,
        None
    };

    struct GPRealNode {
        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> flatten() const = 0;
        virtual void makeIntersect(size_t p, double offsetT, double dt) = 0;
        virtual void applyMemory(GPCorrelationContext ctxt, Vec3d rd) = 0;
        virtual void sampleGrad(int pickId, Vec3d ip, Vec3d rd, Vec3d* points, Derivative* derivs, PathSampleGenerator& sampler, Vec3d& grad) = 0;
        virtual void setGrad(Vec3d grad) = 0;
    };

    struct GPRealNodeCsg : GPRealNode {
        GPRealNodeCsg(std::shared_ptr<GPRealNode> left, std::shared_ptr<GPRealNode> right) : _left(left), _right(right) {}

        std::tuple<double, int> perform_op(double leftSample, double rightSample, int leftId, int rightId) const {
            if (leftSample < rightSample) {
                return { leftSample, leftId };
            }
            else {
                return { rightSample, rightId };
            }
        }
        
        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> flatten() const override {
            auto [lv, li] = _left->flatten();
            auto [rv, ri] = _right->flatten();
            
            Eigen::MatrixXd resV(lv.rows(), lv.cols());
            Eigen::MatrixXi resI(lv.rows(), lv.cols());

            for (size_t c = 0; c < lv.cols(); c++) {
                for (size_t r = 0; r < lv.rows(); r++) {
                    auto [v, id] = perform_op(lv(r,c), rv(r,c), li(r,c), ri(r,c));
                    resV(r,c) = v;
                    resI(r,c) = id;
                }
            }


            return { resV, resI };
        }

        virtual void makeIntersect(size_t p, double offsetT, double dt) override {
            _left->makeIntersect(p, offsetT, dt);
            _right->makeIntersect(p, offsetT, dt);
        }

        virtual void applyMemory(GPCorrelationContext ctxt, Vec3d rd) override {
            _left->applyMemory(ctxt, rd);
            _right->applyMemory(ctxt, rd);
        }

        virtual void sampleGrad(int pickId, Vec3d ip, Vec3d rd, Vec3d* points, Derivative* derivs, PathSampleGenerator& sampler, Vec3d& grad) override {
            _left->sampleGrad(pickId, ip, rd, points, derivs, sampler, grad);
            _right->sampleGrad(pickId, ip, rd, points, derivs, sampler, grad);
        }

        virtual void setGrad(Vec3d grad) override {
            _left->setGrad(grad);
            _right->setGrad(grad);
        }

        std::shared_ptr<GPRealNode> _left, _right;
    };

    struct GPRealNodeValues : GPRealNode {
        GPRealNodeValues(const Eigen::MatrixXd& values, const GaussianProcess* gp) : _values(values), _gp(gp) {}

        Eigen::MatrixXd _values;
        const GaussianProcess* _gp;

        bool _isIntersect = false;
        Vec3d _sampledGrad;

        virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> flatten() const override;
        virtual void makeIntersect(size_t p, double offsetT, double dt) override;
        virtual void sampleGrad(int pickId, Vec3d ip, Vec3d rd, Vec3d* points, Derivative* derivs, PathSampleGenerator& sampler, Vec3d& grad) override;
        virtual void applyMemory(GPCorrelationContext ctxt, Vec3d rd) override;

        virtual void setGrad(Vec3d grad) override {
            _sampledGrad = grad;
        }
    };

    class GPSampleNode : public JsonSerializable {
    public:
        virtual double noIntersectBound(Vec3d p = Vec3d(0.), double q = 0.9999) const = 0;
        virtual double goodStepsize(Vec3d p = Vec3d(0.), double targetCov = 0.95, Vec3d rd = Vec3d(1., 0., 0.)) const = 0;
        virtual double compute_beckmann_roughness(Vec3d p) const = 0;
        virtual double cdf(Vec3d p) const = 0;

        virtual Eigen::VectorXd mean(
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const = 0;

        virtual Vec3d color(Vec3d p) const = 0;
        virtual Vec3d emission(Vec3d p) const = 0;

        virtual std::shared_ptr<GPRealNode> sample_start_value(Vec3d p, PathSampleGenerator& sampler) const = 0;
        virtual std::shared_ptr<GPRealNode> sample(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const = 0;

        virtual std::shared_ptr<GPRealNode> sample_cond(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const GPRealNode* cond_values, const Derivative* cond_derivatives, size_t numCondPts,
            const Vec3d* cond_ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const = 0;
    };

    class GPSampleNodeCSG : public GPSampleNode {
    private:

        std::shared_ptr<GPSampleNode> _left, _right;

    public:

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        double perform_op(double leftSample, double rightSample) const;

        virtual double compute_beckmann_roughness(Vec3d p) const override {
            return _left->compute_beckmann_roughness(p);
        }

        virtual Vec3d color(Vec3d p) const override {
            return _left->color(p);
        }

        virtual Vec3d emission(Vec3d p) const override {
            return _left->emission(p);
        }

        virtual std::shared_ptr<GPRealNode> sample_start_value(Vec3d p, PathSampleGenerator& sampler) const override {
            return std::make_shared<GPRealNodeCsg>(_left->sample_start_value(p, sampler), _right->sample_start_value(p, sampler));
        }

        virtual double noIntersectBound(Vec3d p = Vec3d(0.), double q = 0.9999) const override {
            return max(_left->noIntersectBound(p, q), _right->noIntersectBound(p, q));
        }

        virtual double goodStepsize(Vec3d p = Vec3d(0.), double targetCov = 0.95, Vec3d rd = Vec3d(1., 0., 0.)) const override {
            return min(_left->goodStepsize(p, targetCov, rd), _right->goodStepsize(p, targetCov, rd));
        }

        virtual double cdf(Vec3d p) const override {
            return _left->cdf(p) * _right->cdf(p);
        }


        virtual Eigen::VectorXd mean(
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const override {
            return _left->mean(points, derivative_types, ddirs, deriv_dir, numPts).binaryExpr(
                _right->mean(points, derivative_types, ddirs, deriv_dir, numPts),
                [this](double a, double b) { return perform_op(a, b); }
            );
        }

        virtual std::shared_ptr<GPRealNode> sample(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const override {

            return std::make_shared<GPRealNodeCsg>(_left->sample(
                points, derivative_types, numPts, ddirs,
                constraints, numConstraints, deriv_dir, samples, sampler),
                _right->sample(
                    points, derivative_types, numPts, ddirs,
                    constraints, numConstraints, deriv_dir, samples, sampler));
        }

        virtual std::shared_ptr<GPRealNode> sample_cond(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const GPRealNode* cond_values, const Derivative* cond_derivatives, size_t numCondPts,
            const Vec3d* cond_ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const override {

            auto cond = (GPRealNodeCsg*)(cond_values);

            return std::make_shared<GPRealNodeCsg>(_left->sample_cond(
                points, derivative_types, numPts, ddirs,
                cond_points, cond->_left.get(), cond_derivatives, numCondPts, cond_ddirs,
                constraints, numConstraints, deriv_dir, samples, sampler),
                _right->sample_cond(points, derivative_types, numPts, ddirs,
                    cond_points, cond->_right.get(), cond_derivatives, numCondPts, cond_ddirs,
                    constraints, numConstraints, deriv_dir, samples, sampler));
        }
    };

    class GaussianProcess : public GPSampleNode {
    public:

        GaussianProcess() : _mean(std::make_shared<HomogeneousMean>()), _cov(std::make_shared<SquaredExponentialCovariance>()) { }
        GaussianProcess(std::shared_ptr<MeanFunction> mean, std::shared_ptr<CovarianceFunction> cov) : _mean(mean), _cov(cov) { }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual double compute_beckmann_roughness(Vec3d p) const override {
            return _cov->compute_beckmann_roughness(p);
        }

        virtual Vec3d color(Vec3d p) const override {
            return _mean->color(p);
        }

        virtual Vec3d emission(Vec3d p) const override {
            return _mean->emission(p);
        }

        Vec3d shell_embedding(Vec3d p) const {
            if(_embedCov) {
                return _mean->shell_embedding(p);
            } else {
                return p;
            }
        }

        std::tuple<Eigen::VectorXd, CovMatrix> mean_and_cov(
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const;

        virtual Eigen::VectorXd mean(
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const override;

        Eigen::VectorXd mean_prior(
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const;

        CovMatrix cov(
            const Vec3d* points_a, const Vec3d* points_b,
            const Derivative* dtypes_a, const Derivative* dtypes_b,
            const Vec3d* ddirs_a, const Vec3d* ddirs_b,
            Vec3d deriv_dir, size_t numPtsA, size_t numPtsB) const;

        CovMatrix cov_sym(
            const Vec3d* points_a,
            const Derivative* dtypes_a,
            const Vec3d* ddirs_a,
            Vec3d deriv_dir, size_t numPtsA) const;

        CovMatrix cov_prior(
            const Vec3d* points_a, const Vec3d* points_b,
            const Derivative* dtypes_a, const Derivative* dtypes_b,
            const Vec3d* ddirs_a, const Vec3d* ddirs_b,
            Vec3d deriv_dir, size_t numPtsA, size_t numPtsB) const;

        virtual std::shared_ptr<GPRealNode> sample_start_value(Vec3d p, PathSampleGenerator& sampler) const override;

        MultivariateNormalDistribution create_mvn_cond(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
            const Vec3d* cond_ddirs,
            Vec3d deriv_dir) const;

        virtual std::shared_ptr<GPRealNode> sample(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const override;

        virtual std::shared_ptr<GPRealNode> sample_cond(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const GPRealNode* cond_values, const Derivative* cond_derivatives, size_t numCondPts,
            const Vec3d* cond_ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const override;


        double eval(
            const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            Vec3d deriv_dir) const;


        double eval_cond(
            const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
            const Vec3d* cond_ddirs,
            Vec3d deriv_dir) const;


        void setConditioning(std::vector<Vec3d> globalCondPs,
            std::vector<Derivative> globalCondDerivs,
            std::vector<Vec3d> globalCondDerivDirs,
            std::vector<double> globalCondValues);

        virtual double noIntersectBound(Vec3d p = Vec3d(0.), double q = 0.9999) const override;
        virtual double goodStepsize(Vec3d p = Vec3d(0.), double targetCov = 0.95, Vec3d rd = Vec3d(1., 0., 0.)) const override;
        virtual double cdf(Vec3d p) const override;

    public:

        std::vector<Vec3d> _globalCondPs;
        std::vector<Derivative> _globalCondDerivs;
        std::vector<Vec3d> _globalCondDerivDirs;
        std::vector<double> _globalCondValues;
        Eigen::VectorXd _globalCondPriorMean;

        std::variant<Eigen::BDCSVD<CovMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV>, Eigen::LDLT<CovMatrix>, Eigen::HouseholderQR<CovMatrix>> _globalCondSolver;

        PathPtr _conditioningDataPath;

        std::shared_ptr<MeanFunction> _mean;
        std::shared_ptr<CovarianceFunction> _cov;
        size_t _maxEigenvaluesN = 64;
        float _covEps = 0.f;
        int _id = 0;

        bool _requireCovProjection = false;
        bool _usePseudoInverse = false;
        bool _embedCov = false;
    };
}

#endif /* GAUSSIANPROCESS_HPP_ */