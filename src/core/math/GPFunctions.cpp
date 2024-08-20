#include "GPFunctions.hpp"
#include <math/Mat4f.hpp>
#include <math/MathUtil.hpp>
#include <math/Angle.hpp>

#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <Eigen/SparseQR>
#include <Eigen/Core>

#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/IterativeLinearSolvers>
#include <igl/signed_distance.h>

#include <ccomplex>
#include <random>

namespace Tungsten {

    void CovarianceFunction::loadResources() {}

    double CovarianceFunction::spectral_density(double s) const {
        double max_t = 10;
        double dt = max_t / pow(2, 12);
        double max_w = PI / dt;

        double bin_c = s / max_w * discreteSpectralDensity.size();
        size_t bin = clamp(size_t(bin_c), size_t(0), discreteSpectralDensity.size() - 1);
        size_t n_bin = clamp(size_t(bin_c) + 1, size_t(0), discreteSpectralDensity.size() - 1);

        double bin_frac = bin_c - bin;

        return lerp(discreteSpectralDensity[bin], discreteSpectralDensity[n_bin], bin_frac);
    }


    double CovarianceFunction::sample_spectral_density(PathSampleGenerator& sampler, Vec3d p) const {
        return 0;
    }

    Vec2d CovarianceFunction::sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p) const {
        return Vec2d(0.);
    }

    Vec3d CovarianceFunction::sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p) const {
        return Vec3d(0.);
    }


    FloatD CovarianceFunction::dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const {
        Eigen::Array3d zd = Eigen::Array3d::Zero();
        auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a, b); }, autodiff::along(dirA, zd), at(a, b));
        return covDiff[1];
    }

    FloatD CovarianceFunction::dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const {
        return dcov_da(b, a, dirB);
    }

    FloatDD CovarianceFunction::dcov2_dadb(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const {
        Eigen::Matrix3d hess = autodiff::hessian([&](auto a, auto b) { return cov(a, b); }, wrt(a, b), at(a, b)).block(3, 0, 3, 3);
        double res = dirA.transpose().matrix() * hess * dirB.matrix();
        return res;
    }

    double ProceduralNoise::operator()(Vec3d p) const {
        switch (type) {
        case NoiseType::BottomTop:
            return sqrt(exp(lerp(log(_max),  log(_min), clamp(p.y() * .75, 0., 1.))));
        case NoiseType::BottomTop2:
            return sqrt(exp(lerp(log(_max),  log(_min), clamp(p.y() * 2.0 - 0.25, 0., 1.))));
        case NoiseType::LeftRight:
            return sqrt(exp(lerp(log(_max), log(_min), clamp(p.x()*2 + 1, 0., 1.))));
        case NoiseType::Sandstone:
        {
            p *= 0.3;
            double f = fbm(p + fbm(p + fbm(p, 2), 2), 2);
            Vec3d col = Vec3d(f * 1.9, f * 0.7, f * 0.25);
            col = std::sqrt(col*1.2) - 0.35;
            return lerp(_min, _max, clamp(col.x(), 0., 1.));
        }
        case NoiseType::Rust:
        {
            p *= 2;
            double f = smoothStep(0.4, 0.6, fbm(p + fbm(p*.1, 2)*0.4, 2) - fbm(p*25., 2)*0.1);
            return lerp(_min, _max, clamp(f, 0., 1.));
        }
        }
    }

    Vec3d ProceduralNoiseVec::operator()(Vec3d p) const {
        switch (type) {
        case NoiseType::BottomTop:
            return Vec3d(sqrt(exp(lerp(log(_max),  log(_min), clamp(p.y() * .75, 0., 1.)))));
        case NoiseType::BottomTop2:
            return Vec3d(sqrt(exp(lerp(log(_max),  log(_min), clamp(p.y() * 2.0 - 0.25, 0., 1.)))));
        case NoiseType::LeftRight:
            return Vec3d(sqrt(exp(lerp(log(_max), log(_min), clamp(p.x()*2 + 1, 0., 1.)))));
        case NoiseType::Sandstone:
        {
            p *= 0.3;
            double f = fbm(p + fbm(p + fbm(p, 10), 10), 10);
            Vec3d col = Vec3d(f * 1.9, f * 0.7, f * 0.25);
            col = std::sqrt(col*1.2) - 0.35;
            return  clamp(col * 0.2, Vec3d(0.), Vec3d(1.));
        }
        case NoiseType::Rust:
        {
            p *= 2;
            double f = smoothStep(0.4, 0.6, fbm(p + fbm(p*.1, 10)*0.4, 10) + fbm(p*25., 10)*0.1);
            return lerp(Vec3d(0.278,0.212,0.141), Vec3d(1.), f);
        }
        }
    }

    void NonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
        CovarianceFunction::fromJson(value, scene);

        if (auto cov = value["cov"]) {
            _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
        }

        if (auto variance = value["grid"]) {
            _variance = scene.fetchGrid(variance);
        }
        else if (auto variance = value["variance"]) {
            _variance = scene.fetchGrid(variance);
        }

        if (auto aniso = value["ansio"]) {
            _aniso = scene.fetchGrid(aniso);
        }

        value.getField("offset", _offset);
        value.getField("scale", _scale);
    }

    rapidjson::Value NonstationaryCovariance::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "nonstationary",
            "cov", *_stationaryCov,
            "variance", *_variance,
            "offset", _offset,
            "scale", _scale
        };
    }

    void NonstationaryCovariance::loadResources() {
        CovarianceFunction::loadResources();

        _variance->loadResources();
        _stationaryCov->loadResources();

        if (_aniso) {
            _aniso->loadResources();
        }
    }

    FloatD NonstationaryCovariance::sampleGrid(Vec3Diff a) const {
        FloatD result = 0;
        Vec3f ap = vec_conv<Vec3f>(from_diff(a));
        result[0] = _variance->density(ap);
        return result;

        /**/
        float eps = 0.001f;
        float vals[] = {
            _variance->density(ap + Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap + Vec3f(0.f, eps, 0.f)),
            _variance->density(ap + Vec3f(0.f, 0.f, eps)),
            _variance->density(ap - Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap - Vec3f(0.f, eps, 0.f)),
            _variance->density(ap - Vec3f(0.f, 0.f, eps))
        };
        auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);

        result[1] = grad.dot({ (float)a.x()[1], (float)a.y()[1] , (float)a.z()[1] });
        result[2] = 0; // linear interp
        return result;
    }

    FloatDD NonstationaryCovariance::sampleGrid(Vec3DD a) const {
        FloatDD result = 0;
        Vec3f ap = vec_conv<Vec3f>(from_diff(a));
        result.val = _variance->density(ap);
        return result;

        /**/
        float eps = 0.001f;
        float vals[] = {
            _variance->density(ap + Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap + Vec3f(0.f, eps, 0.f)),
            _variance->density(ap + Vec3f(0.f, 0.f, eps)),
            _variance->density(ap - Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap - Vec3f(0.f, eps, 0.f)),
            _variance->density(ap - Vec3f(0.f, 0.f, eps))
        };
        auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);

        result.grad.val = grad.dot({ a.x().grad.val, a.y().grad.val , a.z().grad.val });
        result.grad.grad = 0; // linear interp
        return result;
    }

   

    FloatD NonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {

        FloatD sigmaA = (sampleGrid(mult(_variance->invNaturalTransform(), a)) + _offset) * _scale;
        FloatD sigmaB = (sampleGrid(mult(_variance->invNaturalTransform(), b)) + _offset) * _scale;
        return sigmaA * sigmaB * _stationaryCov->cov(a, b);

        Mat3Diff anisoA = Mat3Diff::Identity();
        Mat3Diff anisoB = Mat3Diff::Identity();

        FloatD detAnisoA = anisoA.determinant();
        FloatD detAnisoB = anisoB.determinant();

        Mat3Diff anisoABavg = 0.5 * (anisoA + anisoB);
        FloatD detAnisoABavg = anisoABavg.determinant();

        FloatD ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        Vec3Diff d = b - a;
        FloatD dsq = d.transpose() * anisoABavg.inverse() * d;
        return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
    }

    FloatDD NonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {

        auto sigmaA = (sampleGrid(mult(_variance->invNaturalTransform(), a)) + _offset) * _scale;
        auto sigmaB = (sampleGrid(mult(_variance->invNaturalTransform(), b)) + _offset) * _scale;
        return sigmaA * sigmaB * _stationaryCov->cov(a, b);

        auto anisoA = Mat3DD::Identity();
        auto anisoB = Mat3DD::Identity();

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        auto anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        auto d = b - a;
        auto dsq = d.transpose() * anisoABavg.inverse() * d;
        return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
    }

    double NonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
        double sigmaA = (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(a)) + _offset)* _scale;
        double sigmaB = (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(b)) + _offset)* _scale;
        return sigmaA * sigmaB * _stationaryCov->cov(a, b);

        Eigen::Matrix3f anisoA = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f anisoB = Eigen::Matrix3f::Identity();

        double detAnisoA = anisoA.determinant();
        double detAnisoB = anisoB.determinant();

        Eigen::Matrix3f anisoABavg = 0.5 * (anisoA + anisoB);
        double detAnisoABavg = anisoABavg.determinant();

        double ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        Eigen::Vector3f d = vec_conv<Eigen::Vector3f>(b - a);
        double dsq = d.transpose() * anisoABavg.inverse() * d;
        return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
    }

    void NeuralNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
        CovarianceFunction::fromJson(value, scene);

        if (auto path = value["network"]) {
            _path = scene.fetchResource(path);
        }

        value.getField("scale", _scale);
        value.getField("transform", _configTransform);
        _invConfigTransform = _configTransform.invert();
    }

    rapidjson::Value NeuralNonstationaryCovariance::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "nonstationary",
            "network", *_path,
            "scale", _scale,
            "transform", _configTransform
        };
    }

    void NeuralNonstationaryCovariance::loadResources() {
        CovarianceFunction::loadResources();

        std::shared_ptr<JsonDocument> document;
        try {
            document = std::make_shared<JsonDocument>(*_path);
        }
        catch (std::exception& e) {
            std::cerr << e.what() << "\n";
        }

        _nn = std::make_shared<GPNeuralNetwork>();
        _nn->read(*document, _path->absolute().parent());
    }

    FloatD NeuralNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {
        return _nn->cov(mult(_invConfigTransform, a), mult(_invConfigTransform, b)) * _scale;
    }

    FloatDD NeuralNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {
        return _nn->cov(mult(_invConfigTransform, a), mult(_invConfigTransform, b)) * _scale;
    }

    double NeuralNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
        return _nn->cov(mult(_invConfigTransform, a), mult(_invConfigTransform, b)) * _scale;
    }


    void MeanGradNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
        CovarianceFunction::fromJson(value, scene);

        if (auto cov = value["cov"]) {
            _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
        }

        if (auto mean = value["mean"]) {
            _mean = std::dynamic_pointer_cast<MeanFunction>(scene.fetchMeanFunction(mean));
        }

        value.getField("aniso", _aniso);
    }

    rapidjson::Value MeanGradNonstationaryCovariance::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "mg-nonstationary",
            "cov",*_stationaryCov,
            "mean", *_mean,
            "aniso", _aniso,
        };
    }

    void MeanGradNonstationaryCovariance::loadResources() {
        _stationaryCov->loadResources();
    }



    Eigen::Matrix3d MeanGradNonstationaryCovariance::localAniso(Vec3d p) const {
        return compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(p).normalized()),
            vec_conv<Eigen::Vector3d>(_aniso));
    }

    FloatD MeanGradNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {
        Eigen::Matrix3d anisoA = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(a))),
            vec_conv<Eigen::Vector3d>(_aniso));
        Eigen::Matrix3d anisoB = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(b))),
            vec_conv<Eigen::Vector3d>(_aniso));

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        Eigen::Matrix3d anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        auto d = b - a;
        auto dsq = d.transpose() * anisoABavg.inverse() * d;
        return ansioFac * _stationaryCov->cov(dsq);
    }

    FloatDD MeanGradNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {
        Eigen::Matrix3d anisoA = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(a))),
            vec_conv<Eigen::Vector3d>(_aniso));
        Eigen::Matrix3d anisoB = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(b))),
            vec_conv<Eigen::Vector3d>(_aniso));

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        Eigen::Matrix3d anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        auto d = b - a;
        auto dsq = d.transpose() * anisoABavg.inverse() * d;
        return ansioFac * _stationaryCov->cov(dsq);
    }

    double MeanGradNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
        auto anisoA = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(a)),
            vec_conv<Eigen::Vector3d>(_aniso));
        auto anisoB = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(b)),
            vec_conv<Eigen::Vector3d>(_aniso));

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        Eigen::Matrix3d anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        float ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        Eigen::Vector3d d = vec_conv<Eigen::Vector3d>(b - a);
        double dsq = d.transpose() * anisoABavg.inverse() * d;
        return ansioFac * _stationaryCov->cov(dsq);
    }

    void TabulatedMean::fromJson(JsonPtr value, const Scene& scene) {
        MeanFunction::fromJson(value, scene);

        if (auto grid = value["grid"]) {
            _grid = scene.fetchGrid(grid);
        }

        value.getField("offset", _offset);
        value.getField("scale", _scale);
        value.getField("volume", _isVolume);
    }

    rapidjson::Value TabulatedMean::toJson(Allocator& allocator) const {
        return JsonObject{ MeanFunction::toJson(allocator), allocator,
            "type", "tabulated",
            "grid", *_grid,
            "offset", _offset,
            "scale", _scale,
            "volume", _isVolume
        };
    }


    double TabulatedMean::mean(Vec3d a) const {
        Vec3f p = _grid->invNaturalTransform() * vec_conv<Vec3f>(a);
        double res = _grid->density(p);
        if(_isVolume) {
            res = -log(max(0.0001, res));
        }
        return (res + _offset) * _scale;
    }

    Vec3d TabulatedMean::emission(Vec3d a) const {
        Vec3f p = _grid->invNaturalTransform() * vec_conv<Vec3f>(a);
        return Vec3d(_grid->emission(p));
    }

    Vec3d TabulatedMean::dmean_da(Vec3d a) const {
        double eps = 0.001;
        double vals[] = {
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps)),
            mean(a)
        };

        auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[3], vals[2] - vals[3]) / eps;
        return grad;
        /*Vec3d p = _grid->invNaturalTransform() * a;
        return _scale* _grid->naturalTransform().transformVector(_grid->gradient(p));*/
    }

    void TabulatedMean::loadResources() {
        _grid->loadResources();
    }


    void NeuralMean::fromJson(JsonPtr value, const Scene& scene) {
        MeanFunction::fromJson(value, scene);

        if (auto path = value["network"]) _path = scene.fetchResource(path);

        value.getField("offset", _offset);
        value.getField("scale", _scale);
        value.getField("transform", _configTransform);

        _invConfigTransform = _configTransform.invert();
    }

    rapidjson::Value NeuralMean::toJson(Allocator& allocator) const {
        return JsonObject{ MeanFunction::toJson(allocator), allocator,
            "type", "neural",
            "network", *_path,
            "offset", _offset,
            "scale", _scale,
            "transform", _configTransform
        };
    }


    double NeuralMean::mean(Vec3d a) const {
        return (_nn->mean(mult(_invConfigTransform, a)) + _offset) * _scale;
    }

    Vec3d NeuralMean::dmean_da(Vec3d a) const {
        double eps = 0.001;
        double vals[] = {
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps)),
            mean(a)
        };

        auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[3], vals[2] - vals[3]) / eps;
        return grad;
    }

    void NeuralMean::loadResources() {
        std::shared_ptr<JsonDocument> document;
        try {
            document = std::make_shared<JsonDocument>(*_path);
        }
        catch (std::exception& e) {
            std::cerr << e.what() << "\n";
        }

        _nn = std::make_shared<GPNeuralNetwork>();
        _nn->read(*document, _path->absolute().parent());
    }

    void ProceduralMean::fromJson(JsonPtr value, const Scene& scene) {
        MeanFunction::fromJson(value, scene);
        value.getField("transform", _configTransform);
        _invConfigTransform = _configTransform.invert();

        std::string fnString = "knob";
        if (value.getField("func", fnString)) {
            _f = std::make_shared<ProceduralSdf>(SdfFunctions::stringToFunction(fnString));
        }
        else if(auto f = value["f"]) {
            _f = scene.fetchProceduralScalar(f);
        }

        value.getField("min", _min);
        value.getField("scale", _scale);
        value.getField("offset", _offset);
    }

    rapidjson::Value ProceduralMean::toJson(Allocator& allocator) const {
        return  JsonObject{ MeanFunction::toJson(allocator), allocator,
            "type", "procedural",
            "f", *_f,
            "transform", _configTransform,
            "min", _min,
            "scale", _scale,
            "scale", _offset,
        };
    }

    double ProceduralMean::mean(Vec3d a) const {
        auto p = vec_conv<Vec3f>(a);
        p = _invConfigTransform.transformPoint(p);
        float m = (*_f)(a);
        m *= _scale;
        return max(_min, m + _offset);
    }

    Vec3d ProceduralMean::dmean_da(Vec3d a) const {
        double eps = 0.001f;
        double vals[] = {
            mean(a),
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps))
        };
        return Vec3d(vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]) / eps;
    }


    void MeshSdfMean::fromJson(JsonPtr value, const Scene& scene) {
        MeanFunction::fromJson(value, scene);
        if (auto path = value["file"]) _path = scene.fetchResource(path);
        value.getField("transform", _configTransform);
        value.getField("signed", _signed);
        value.getField("min", _min);
        _invConfigTransform = _configTransform.invert();
    }

    rapidjson::Value MeshSdfMean::toJson(Allocator& allocator) const {
        return JsonObject{ MeanFunction::toJson(allocator), allocator,
            "type", "mesh",
            "file", *_path,
            "transform", _configTransform,
            "signed", _signed,
            "min", _min,
        };
    }


    double MeshSdfMean::mean(Vec3d a) const {
        // perform a closest point query
        Eigen::MatrixXd V_vis(1, 3);
        V_vis(0, 0) = a.x();
        V_vis(1, 0) = a.y();
        V_vis(2, 0) = a.z();

        Eigen::VectorXd S_vis;
        igl::signed_distance_fast_winding_number(V_vis, V, F, tree, fwn_bvh, S_vis);

        return max((double)S_vis(0), _min);
    }

    Vec3d MeshSdfMean::color(Vec3d a) const {
        Eigen::RowVector3d P = vec_conv<Eigen::RowVector3d>(a);

        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::RowVector3d closest_point;
        tree.squared_distance(V, F, P, sqrD, I, closest_point);

        Eigen::VectorXi closestFace = F.row(I[0]);

        Eigen::RowVector3d 
            vA = V.row(closestFace[0]), 
            vB = V.row(closestFace[1]), 
            vC = V.row(closestFace[2]);

        Eigen::RowVector3d L;
        igl::barycentric_coordinates(
            closest_point,
            vA, vB, vC,
            L);

        Vec3f colA = _colors[closestFace[0]];
        Vec3f colB = _colors[closestFace[1]];
        Vec3f colC = _colors[closestFace[2]];

        Vec3d result = vec_conv<Vec3d>(colA * L[0] + colB * L[1] + colC * L[2]);

        if (std::isinf(result) || std::isnan(result) || (colA + colB + colC).sum() < 0.05f) {
            return vec_conv<Vec3d>(colA + colB + colC) / 3;
        }

        return result;
    }

    Vec3d MeshSdfMean::shell_embedding(Vec3d a) const {
        //return a;
        Eigen::RowVector3d P = vec_conv<Eigen::RowVector3d>(a);

        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::RowVector3d closest_point;
        tree.squared_distance(V, F, P, sqrD, I, closest_point);

        Eigen::VectorXi closestFace = F.row(I[0]);

        Eigen::RowVector3d
            vA = V.row(closestFace[0]),
            vB = V.row(closestFace[1]),
            vC = V.row(closestFace[2]);

        Eigen::RowVector3d L;
        igl::barycentric_coordinates(
            closest_point,
            vA, vB, vC,
            L);

        Vec2f uvA = _uvs[closestFace[0]];
        Vec2f uvB = _uvs[closestFace[1]];
        Vec2f uvC = _uvs[closestFace[2]];

        Vec2d uv = Vec2d(uvA * L[0] + uvB * L[1] + uvC * L[2]) * _bounds.diagonal().length();

        double w = igl::fast_winding_number(fwn_bvh, 2, P);
        //0.5 is on surface
        double dist = sqrt(sqrD(0)) * (1. - 2. * std::abs(w));

        return Vec3d(uv.x(), uv.y(), dist);
    }

    Vec3d MeshSdfMean::dmean_da(Vec3d a) const {
        double eps = 0.001f;
        double vals[] = {
            mean(a),
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps))
        };

        return Vec3d(vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]) / eps;
    }

    void MeshSdfMean::loadResources() {

        std::vector<Vertex> _verts;
        std::vector<TriangleI> _tris;

        _bounds = Box3d();

        if (_path && MeshIO::load(*_path, _verts, _tris)) {

            V.resize(_verts.size(), 3);
            _colors.resize(_verts.size());
            _uvs.resize(_verts.size());

            for (int i = 0; i < _verts.size(); i++) {
                Vec3f tpos = _configTransform * _verts[i].pos();
                V(i, 0) = tpos.x();
                V(i, 1) = tpos.y();
                V(i, 2) = tpos.z();

                _colors[i] = _verts[i].color();

                if (_colors[i].sum() < 0.001) {
                    _colors[i] = Vec3f(0.215684f, 0.262744f, 0.031373f);
                }

                _uvs[i] = _verts[i].uv();

                //Vec3f tnorm = _configTransform.transformVector(_verts[i].normal());
             
                _bounds.grow(vec_conv<Vec3d>(tpos));
            }

            F.resize(_tris.size(), 3);

            // specify the triangle indices
            for (int i = 0; i < _tris.size(); i++) {
                F(i, 0) = _tris[i].v0;
                F(i, 1) = _tris[i].v1;
                F(i, 2) = _tris[i].v2;
            }

            tree.init(V, F);
            igl::fast_winding_number(V, F, 2, fwn_bvh);
        }
        else {
            FAIL("Failed to load mesh.");
        }
    }

   
}