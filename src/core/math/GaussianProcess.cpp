#include "GaussianProcess.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "ziggurat_constants.h"
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <Eigen/SparseQR>
#include <Eigen/Core>

#include <Spectra/MatOp/SparseGenMatProd.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/IterativeLinearSolvers>

namespace Tungsten {

#ifdef SPARSE_COV
//#define SPARSE_SOLVE
#endif

void GPSampleNodeCSG::fromJson(JsonPtr value, const Scene& scene) {
    GPSampleNode::fromJson(value, scene);

    if (auto left = value["left"])
        _left = scene.fetchGaussianProcess(left);
    if (auto right = value["right"])
        _right = scene.fetchGaussianProcess(right);
}

rapidjson::Value GPSampleNodeCSG::toJson(Allocator& allocator) const {
    return JsonObject{ GPSampleNode::toJson(allocator), allocator,
        "type", "csg",
        "left", *_left,
        "right", *_right,
    };
}

void GPSampleNodeCSG::loadResources() {
    _left->loadResources();
    _right->loadResources();
}

double GPSampleNodeCSG::perform_op(double leftSample, double rightSample) const {
    return min(leftSample, rightSample);
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> GPRealNodeValues::flatten() const {
    return { _values, Eigen::MatrixXi::Ones(_values.rows(), _values.cols()) * _gp->_id};
}

void GPRealNodeValues::makeIntersect(size_t p, double offsetT, double dt) {
    _values.conservativeResize(p + 2, Eigen::NoChange);
    auto prevV = _values(p - 1, 0);
    auto currV = _values(p, 0);

    // Value at intersection point
    _values(p, 0) = lerp(prevV, currV, offsetT);
    // Derivative at intersection point;
    _values(p + 1, 0) = (prevV - currV) / dt;

    _isIntersect = true;
}

void GPRealNodeValues::sampleGrad(int pickId, Vec3d ip, Vec3d rd, Vec3d* points, Derivative* derivs, PathSampleGenerator& sampler, Vec3d& grad) {
    std::array<Vec3d, 3> gradPs{ ip, ip, ip };
    std::array<Derivative, 3> gradDerivs{ Derivative::First, Derivative::First, Derivative::First };

    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(rd));

    if (_isIntersect) {
        std::array<Vec3d, 2> gradDirs{
            vec_conv<Vec3d>(frame.tangent),
            vec_conv<Vec3d>(frame.bitangent)
        };

        auto [gradSamples, gradId] = _gp->sample_cond(
            gradPs.data(), gradDerivs.data(), gradDirs.size(), gradDirs.data(),
            points, this, derivs, _values.rows(), nullptr,
            nullptr, 0,
            rd, 1, sampler)->flatten();

        _sampledGrad = vec_conv<Vec3d>(frame.toGlobal({
            gradSamples(0,0), gradSamples(1,0), _values(_values.rows() - 1,0)
            }));

    }
    else {
        std::array<Vec3d, 3> gradDirs{
            vec_conv<Vec3d>(frame.tangent),
            vec_conv<Vec3d>(frame.bitangent),
            vec_conv<Vec3d>(frame.normal)
        };

        auto [gradSamples, gradId] = _gp->sample_cond(
            gradPs.data(), gradDerivs.data(), gradDirs.size(), gradDirs.data(),
            points, this, derivs, _values.rows(), nullptr,
            nullptr, 0,
            rd, 1, sampler)->flatten();

        _sampledGrad = vec_conv<Vec3d>(frame.toGlobal({
            gradSamples(0,0), gradSamples(1,0), gradSamples(2,0)
            }));
    }

    if (_gp->_id == pickId) {
        grad = _sampledGrad;
    }
}

void GPRealNodeValues::applyMemory(GPCorrelationContext ctxt, Vec3d rd) {
    switch (ctxt) {
    case GPCorrelationContext::None: {
        _values = Eigen::MatrixXd(0, _values.cols());
        break;
    }
    case GPCorrelationContext::Dori: {
        auto newValues = Eigen::VectorXd(1);
        if (_isIntersect)
            newValues(0) = _values(_values.size() - 2, 0);
        else
            newValues(0) = _values(_values.size() - 1, 0);
        _values = newValues;
        break;
    }
    case GPCorrelationContext::Goldfish: {
        auto newValues = Eigen::VectorXd(2);
        if (_isIntersect)
            newValues(0) = _values(_values.size() - 2, 0);
        else
            newValues(0) = _values(_values.size() - 1, 0);
       
        newValues(1) = _sampledGrad.dot(rd);
        _values = newValues;
        break;
    }
    case GPCorrelationContext::Elephant: {
        _values.conservativeResize(_values.size() + 1, Eigen::NoChange);
        _values.row(_values.size() - 1).array() = _sampledGrad.dot(rd);
        break;
    }
    }
}


void GaussianProcess::fromJson(JsonPtr value, const Scene& scene) {
    GPSampleNode::fromJson(value, scene);

    if (auto mean = value["mean"])
        _mean = scene.fetchMeanFunction(mean);
    if (auto cov = value["covariance"])
        _cov = scene.fetchCovarianceFunction(cov);

    value.getField("max_num_eigenvalues", _maxEigenvaluesN);
    value.getField("covariance_epsilon", _covEps);
    value.getField("id", _id);
    value.getField("project_cov", _requireCovProjection);
    value.getField("pseudo_inverse", _usePseudoInverse);
    value.getField("embed_cov", _embedCov);

    if (auto conditioningDataPath = value["conditioning_data"])
        _conditioningDataPath = scene.fetchResource(conditioningDataPath);

}

rapidjson::Value GaussianProcess::toJson(Allocator& allocator) const {
    auto obj = JsonObject{ GPSampleNode::toJson(allocator), allocator,
        "type", "standard",
        "mean", *_mean,
        "covariance", *_cov,
        "max_num_eigenvalues", _maxEigenvaluesN,
        "covariance_epsilon", _covEps,
        "id", _id,
        "project_cov", _requireCovProjection,
        "pseudo_inverse", _usePseudoInverse,
        "embed_cov", _embedCov
    };

    if (_conditioningDataPath) {
        obj.add("conditioning_data", *_conditioningDataPath);
    }

    return obj;
}

void GaussianProcess::loadResources() {
    _mean->loadResources();
    _cov->loadResources();

    std::vector<Vertex> verts;
    std::vector<TriangleI> tris;
    if (_conditioningDataPath && MeshIO::load(*_conditioningDataPath, verts, tris)) {
        std::unordered_set<Vertex> unique_verts;
        for (const auto& v : verts) {
            unique_verts.insert(v);
        }


        for (const auto& v : unique_verts) {
            _globalCondPs.push_back(vec_conv<Vec3d>(v.pos()));
            _globalCondDerivs.push_back(Derivative::None);
            _globalCondDerivDirs.push_back(vec_conv<Vec3d>(v.normal()));
            _globalCondValues.push_back(0);

            _globalCondPs.push_back(vec_conv<Vec3d>(v.pos()));
            _globalCondDerivs.push_back(Derivative::First);
            _globalCondDerivDirs.push_back(vec_conv<Vec3d>(v.normal()));
            _globalCondValues.push_back(1);
        }

        setConditioning(_globalCondPs, _globalCondDerivs, _globalCondDerivDirs, _globalCondValues);
    }

    _requireCovProjection |= _cov->requireProjection();
}

void GaussianProcess::setConditioning(
    std::vector<Vec3d> globalCondPs,
    std::vector<Derivative> globalCondDerivs,
    std::vector<Vec3d> globalCondDerivDirs,
    std::vector<double> globalCondValues) {

    CovMatrix s11 = cov_prior(
        globalCondPs.data(), globalCondPs.data(),
        globalCondDerivs.data(), globalCondDerivs.data(),
        globalCondDerivDirs.data(), globalCondDerivDirs.data(),
        Vec3d(0.), globalCondPs.size(), globalCondPs.size());

    //s11.diagonal().array() += 0.001;

    //s11 = project_to_psd(s11);

    Eigen::Map<const Eigen::VectorXd> cond_values_view(globalCondValues.data(), globalCondValues.size());
    _globalCondPriorMean = cond_values_view - mean_prior(globalCondPs.data(), globalCondDerivs.data(), globalCondDerivDirs.data(), Vec3d(0.), globalCondPs.size());

    _globalCondPs = globalCondPs;
    _globalCondDerivs = globalCondDerivs;
    _globalCondDerivDirs = globalCondDerivDirs;
    _globalCondValues = globalCondValues;

    CovMatrix solved;

    bool succesfullSolve = false;
    if (true || s11.rows() <= 64) {
        auto lltSolver = Eigen::LDLT<CovMatrix>(s11.triangularView<Eigen::Lower>());

        if (lltSolver.info() == Eigen::ComputationInfo::Success) {
            succesfullSolve = true;
            _globalCondSolver = lltSolver;
            std::cout << "Using LDLT solver for global conditioning.\n";
        }
    }

#if 0
    if (!succesfullSolve) {
        auto solver = Eigen::HouseholderQR<CovMatrix>();
        solver.compute(s11.triangularView<Eigen::Lower>());

        _globalCondSolver = solver;
        std::cout << "Using HouseholderQR solver for global conditioning.\n";
        succesfullSolve = true;

        /*if (solver. == Eigen::ComputationInfo::Success) {
            succesfullSolve = true;
            _globalCondSolver = bdcsvdSolver;
            std::cout << "Using BDCSVD solver for global conditioning.\n";
        }*/
    }
#endif

    if (!succesfullSolve) {
        auto bdcsvdSolver = Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>();
        bdcsvdSolver.compute(s11.triangularView<Eigen::Lower>());

        if (bdcsvdSolver.info() == Eigen::ComputationInfo::Success) {
            succesfullSolve = true;
            _globalCondSolver = bdcsvdSolver;
            std::cout << "Using BDCSVD solver for global conditioning.\n";
        }
    }

    if (!succesfullSolve) {
        FAIL("Global conditioning decomposition failed!\n");
    }
}

std::tuple<Eigen::VectorXd, CovMatrix> GaussianProcess::mean_and_cov(
    const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
    Vec3d deriv_dir, size_t numPts) const {

    Eigen::VectorXd ps_mean(numPts);
    CovMatrix ps_cov(numPts, numPts);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(min(numPts * numPts / 10, (size_t)10000));
#endif

    for (size_t i = 0; i < numPts; i++) {
        const Vec3d& ddir_a = ddirs ? ddirs[i] : deriv_dir;
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], ddir_a);

        for (size_t j = 0; j <= i; j++) {
            const Vec3d& ddir_b = ddirs ? ddirs[j] : deriv_dir;
            double cov_ij = (*_cov)(derivative_types[i], derivative_types[j], shell_embedding(points[i]), shell_embedding(points[j]), ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
            }
#else
            ps_cov(i, j) = ps_cov(j, i) = cov_ij;
#endif
        }
    }

#ifdef SPARSE_COV
    ps_cov.setFromTriplets(tripletList.begin(), tripletList.end());
    ps_cov.makeCompressed();
#endif

    if (_globalCondPs.size() != 0) {
        CovMatrix s12 = cov_prior(
            _globalCondPs.data(), points,
            _globalCondDerivs.data(), derivative_types,
            _globalCondDerivDirs.data(), ddirs,
            deriv_dir, _globalCondPs.size(), numPts);


        CovMatrix solved = std::visit([&s12](auto&& solver) -> CovMatrix { return solver.solve(s12).transpose(); }, _globalCondSolver);

        ps_mean += (solved * _globalCondPriorMean);
        ps_cov -= (solved * s12);
    }

    if (_requireCovProjection) {
        ps_cov = project_to_psd(ps_cov);
    }

    return { ps_mean, ps_cov };
}

Eigen::VectorXd GaussianProcess::mean_prior(
    const Vec3d * points, const Derivative * derivative_types, const Vec3d * ddirs,
    Vec3d deriv_dir, size_t numPts) const {

    Eigen::VectorXd ps_mean(numPts);
    for (size_t i = 0; i < numPts; i++) {
        const Vec3d& ddir = ddirs ? ddirs[i] : deriv_dir;
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], ddir);
    }
    return ps_mean;
}

Eigen::VectorXd GaussianProcess::mean(
    const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
    Vec3d deriv_dir, size_t numPts) const {

    Eigen::VectorXd ps_mean = mean_prior(points, derivative_types, ddirs, deriv_dir, numPts);

    if (_globalCondPs.size() == 0) {
        return ps_mean;
    }
    else {
        CovMatrix s12 = cov_prior(
            _globalCondPs.data(), points,
            _globalCondDerivs.data(), derivative_types,
            _globalCondDerivDirs.data(), ddirs,
            deriv_dir, _globalCondPs.size(), numPts);

        CovMatrix solved = std::visit([&s12](auto&& solver) -> CovMatrix { return solver.solve(s12).transpose(); }, _globalCondSolver);

        return ps_mean + (solved * _globalCondPriorMean);
    }
}

CovMatrix GaussianProcess::cov_prior(
    const Vec3d* points_a, const Vec3d* points_b,
    const Derivative* dtypes_a, const Derivative* dtypes_b,
    const Vec3d* ddirs_a, const Vec3d* ddirs_b,
    Vec3d deriv_dir, size_t numPtsA, size_t numPtsB) const {

    CovMatrix ps_cov(numPtsA, numPtsB);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(min(numPtsA * numPtsB / 10, (size_t)10000));
#endif


    for (size_t i = 0; i < numPtsA; i++) {
        const Vec3d& ddir_a = ddirs_a ? ddirs_a[i] : deriv_dir;
        for (size_t j = 0; j < numPtsB; j++) {
            const Vec3d& ddir_b = ddirs_b ? ddirs_b[j] : deriv_dir;

            double cov_ij = (*_cov)(dtypes_a[i], dtypes_b[j], shell_embedding(points_a[i]), shell_embedding(points_b[j]), ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
            }
#else
            ps_cov(i, j) = cov_ij;
#endif
        }
    }

#ifdef SPARSE_COV
    ps_cov.setFromTriplets(tripletList.begin(), tripletList.end());
    ps_cov.makeCompressed();
#endif

    return ps_cov;
}


CovMatrix GaussianProcess::cov(
    const Vec3d* points_a, const Vec3d* points_b, 
    const Derivative* dtypes_a, const Derivative* dtypes_b,
    const Vec3d* ddirs_a, const Vec3d* ddirs_b,
    Vec3d deriv_dir, size_t numPtsA, size_t numPtsB) const {
    CovMatrix ps_cov = cov_prior(
        points_a, points_b,
        dtypes_a, dtypes_b,
        ddirs_a, ddirs_b,
        deriv_dir, numPtsA, numPtsB);

    if (_globalCondPs.size() == 0) {
        return ps_cov;
    }
    else {

        CovMatrix sCB = cov_prior(
            _globalCondPs.data(), points_b,
            _globalCondDerivs.data(), dtypes_b,
            _globalCondDerivDirs.data(), ddirs_b,
            deriv_dir, _globalCondPs.size(), numPtsB);
        CovMatrix solvedSCB = std::visit([&sCB](auto&& solver) -> CovMatrix { return solver.solve(sCB); }, _globalCondSolver);

        CovMatrix sCA = cov_prior(
            _globalCondPs.data(), points_a,
            _globalCondDerivs.data(), dtypes_a,
            _globalCondDerivDirs.data(), ddirs_a,
            deriv_dir, _globalCondPs.size(), numPtsA);

        return ps_cov - (sCA.transpose() * solvedSCB);
    }
}

CovMatrix GaussianProcess::cov_sym(
    const Vec3d* points_a,
    const Derivative* dtypes_a,
    const Vec3d* ddirs_a,
    Vec3d deriv_dir, size_t numPtsA) const {
    CovMatrix ps_cov(numPtsA, numPtsA);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(min(numPtsA * numPtsB / 10, (size_t)10000));
#endif


    for (size_t i = 0; i < numPtsA; i++) {
        const Vec3d& ddir_a = ddirs_a ? ddirs_a[i] : deriv_dir;
        for (size_t j = 0; j <= i; j++) {
            const Vec3d& ddir_b = ddirs_a ? ddirs_a[j] : deriv_dir;

            double cov_ij = (*_cov)(dtypes_a[i], dtypes_a[j], shell_embedding(points_a[i]), shell_embedding(points_a[j]), ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
                tripletList.push_back(Eigen::Triplet<double>(j, i, cov_ij));
            }
#else
            ps_cov(j, i) = ps_cov(i, j) = cov_ij;
#endif
        }
    }

#ifdef SPARSE_COV
    ps_cov.setFromTriplets(tripletList.begin(), tripletList.end());
    ps_cov.makeCompressed();
#endif

    
    if(_globalCondPs.size() != 0) {
        CovMatrix s12 = cov_prior(
            _globalCondPs.data(), points_a,
            _globalCondDerivs.data(), dtypes_a,
            _globalCondDerivDirs.data(), ddirs_a,
            deriv_dir, _globalCondPs.size(), numPtsA);

        CovMatrix solved = std::visit([&s12](auto&& solver) -> CovMatrix { return solver.solve(s12).transpose(); }, _globalCondSolver);

        ps_cov = ps_cov - (solved * s12);
    }

    if (_requireCovProjection) {
        ps_cov = project_to_psd(ps_cov);
    }

    return ps_cov;
}

std::shared_ptr<GPRealNode> GaussianProcess::sample_start_value(Vec3d p, PathSampleGenerator& sampler) const {
    auto deriv = Derivative::None;
    auto [mean, cov] = mean_and_cov(&p, &deriv, nullptr, Vec3d(0.), 1);

    double m = mean(0);
    double sigma = sqrt(cov(0,0));

    Eigen::VectorXd vals(1);

    vals(0) = max(0., rand_truncated_normal(m, sigma, 0, sampler));

    return std::make_shared<GPRealNodeValues>(vals, this);
}


std::shared_ptr<GPRealNode> GaussianProcess::sample(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const {

    auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, deriv_dirs, deriv_dir, numPts);
    auto mvn = MultivariateNormalDistribution(ps_mean, ps_cov);
    return std::make_shared<GPRealNodeValues>(mvn.sample(constraints, numConstraints, samples, sampler), this);
}

std::shared_ptr<GPRealNode> GaussianProcess::sample_cond(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* deriv_dirs,
    const Vec3d* cond_points, const GPRealNode* cond_values, const Derivative* cond_derivatives, size_t numCondPts,
    const Vec3d* cond_deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const {

    auto real = (const GPRealNodeValues*)cond_values;

    auto mvn = create_mvn_cond(points, derivative_types, numPts, deriv_dirs,
        cond_points, real->_values.data(), cond_derivatives, numCondPts, cond_deriv_dirs,
        deriv_dir);

    return std::make_shared<GPRealNodeValues>(mvn.sample(constraints, numConstraints, samples, sampler), this);
}


double GaussianProcess::eval(
    const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
    const Vec3d* ddirs,
    Vec3d deriv_dir) const {

    Eigen::Map<const Eigen::VectorXd> eval_values_View(values, numPts);
    auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, ddirs, deriv_dir, numPts);
    auto mvn = MultivariateNormalDistribution(ps_mean, ps_cov);
    return mvn.eval(eval_values_View);
}

double _eigvalsh_to_eps(const Eigen::VectorXd& s) {
    return 1e6 * DBL_EPSILON * s.cwiseAbs().maxCoeff();
}

Eigen::VectorXd _pinv_1d(const Eigen::VectorXd& v, double eps = 1e-5) {
    return v.cwiseAbs().cwiseLessOrEqual(eps).select(
        0., v.cwiseInverse()
    );
}


CovMatrix pseudo_inverse(const CovMatrix& a) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(a);
    double eps = _eigvalsh_to_eps(eigs.eigenvalues());
    Eigen::VectorXd s_pinv = _pinv_1d(eigs.eigenvalues(), eps);
    Eigen::MatrixXd U = eigs.eigenvectors() * s_pinv.cwiseSqrt().asDiagonal();
    return U * U.transpose();
}

MultivariateNormalDistribution GaussianProcess::create_mvn_cond(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* ddirs,
    const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3d* cond_ddirs,
    Vec3d deriv_dir) const {

    if (numCondPts == 0) {
        auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, ddirs, deriv_dir, numPts);
        return MultivariateNormalDistribution(ps_mean, ps_cov);
    }

    CovMatrix s11 = cov_sym(
        cond_points,
        cond_derivative_types,
        cond_ddirs,
        deriv_dir, numCondPts);

    CovMatrix s12 = cov(
        cond_points, points,
        cond_derivative_types, derivative_types,
        cond_ddirs, ddirs,
        deriv_dir, numCondPts, numPts);

    
    CovMatrix solved;
    if (_usePseudoInverse) {
        solved = (pseudo_inverse(s11) * s12).transpose();
    }
    else {
        bool succesfullSolve = false;
        if (true || s11.rows() <= 64) {
#ifdef SPARSE_COV
            Eigen::SimplicialLDLT<CovMatrix> solver(s11);
#else
            Eigen::LDLT<CovMatrix> solver(s11.triangularView<Eigen::Lower>());
#endif
            if (solver.info() == Eigen::ComputationInfo::Success && solver.isPositive()) {
                solved = solver.solve(s12).transpose();
                if (solver.info() == Eigen::ComputationInfo::Success) {
                    succesfullSolve = true;
                }
                else {
                    std::cerr << "Conditioning solving failed (LDLT)!\n";
                }
            }
        }


        if (!succesfullSolve) {
            Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> solver;
            solver.compute(s11.triangularView<Eigen::Lower>());

            if (solver.info() != Eigen::ComputationInfo::Success) {
                std::cerr << "Conditioning decomposition failed (BDCSVD)!\n";
            }

#ifdef SPARSE_COV
            Eigen::MatrixXd solvedDense = solver.solve(s12.toDense()).transpose();
            solved = solvedDense.sparseView();
#else
            solved = solver.solve(s12).transpose();
#endif
            if (solver.info() != Eigen::ComputationInfo::Success) {
                std::cerr << "Conditioning solving failed (BDCSVD)!\n";
            }
        }
    }

    Eigen::Map<const Eigen::VectorXd> cond_values_view(cond_values, numCondPts);
    Eigen::VectorXd m2 = mean(points, derivative_types, ddirs, deriv_dir, numPts) + (solved * (cond_values_view - mean(cond_points, cond_derivative_types, cond_ddirs, deriv_dir, numCondPts)));

    CovMatrix s22 = cov_sym(
        points,
        derivative_types,
        ddirs,
        deriv_dir, numPts);

    CovMatrix s2 = s22 - (solved * s12);

    if (_requireCovProjection) {
        s2 = project_to_psd(s2);
    }

    return MultivariateNormalDistribution(m2, s2);
}

double GaussianProcess::eval_cond(
    const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
    const Vec3d* ddirs,
    const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3d* cond_ddirs,
    Vec3d deriv_dir) const {

    Eigen::Map<const Eigen::VectorXd> eval_values_View(values, numPts);
    auto mvn = create_mvn_cond(points, derivative_types, numPts, ddirs, 
        cond_points, cond_values, cond_derivative_types, numCondPts, cond_ddirs,
        deriv_dir);

    return mvn.eval(eval_values_View);
}

double GaussianProcess::noIntersectBound(Vec3d p, double q) const
{
    double stddev = sqrt((*_cov)(Derivative::None, Derivative::None, shell_embedding(p), shell_embedding(p), Vec3d(0.), Vec3d(0.)));
    return stddev * sqrt(2.) * boost::math::erf_inv(2 * q - 1);
}

double GaussianProcess::cdf(Vec3d p) const
{
    auto deriv = Derivative::None;
    double stddev = sqrt(cov_sym(&p, &deriv, nullptr, Vec3d(), 1)(0,0));
    double mu = mean(&p, &deriv, nullptr, Vec3d(), 1)(0);
    return 0.5 * (1 + boost::math::erf( (0 - mu) / (stddev * sqrt(2))));
}

double GaussianProcess::goodStepsize(Vec3d p, double targetCov, Vec3d rd) const
{
    double sigma = (*_cov)(Derivative::None, Derivative::None, shell_embedding(p), shell_embedding(p), Vec3d(0.), Vec3d(0.));

    double stepsize_lb = 0;
    double stepsize_ub = 2;

    double stepsize_avg = (stepsize_lb + stepsize_ub) * 0.5;
    double cov = (*_cov)(Derivative::None, Derivative::None, shell_embedding(p), shell_embedding(p + rd * stepsize_avg), Vec3d(0.), Vec3d(0.));

    size_t it = 0;
    while (std::abs(cov - targetCov) > 0.00000000001 && it++ < 100) {
        stepsize_avg = (stepsize_lb + stepsize_ub) * 0.5;
        if (cov > targetCov) {
            stepsize_lb = stepsize_avg;
        }
        else {
            stepsize_ub = stepsize_avg;
        }
        cov = (*_cov)(Derivative::None, Derivative::None, shell_embedding(p), shell_embedding(p + rd * stepsize_avg), Vec3d(0.), Vec3d(0.)) / sigma;
    } 

    return stepsize_avg;
}


}