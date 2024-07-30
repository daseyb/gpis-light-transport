#include "GaussianProcessFactory.hpp"

#include "GaussianProcess.hpp"
#include "GPNeuralNetwork.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(GaussianProcessFactory, "gaussian_process", ({
    {"standard", std::make_shared<GaussianProcess>},
    {"csg", std::make_shared<GPSampleNodeCSG>},
}))


DEFINE_STRINGABLE_ENUM(MeanFunctionFactory, "mean", ({
    {"homogeneous", std::make_shared<HomogeneousMean>},
    {"spherical", std::make_shared<SphericalMean>},
    {"linear", std::make_shared<LinearMean>},
    {"tabulated", std::make_shared<TabulatedMean>},
    {"mesh", std::make_shared<MeshSdfMean>},
    {"procedural", std::make_shared<ProceduralMean>},
    {"neural", std::make_shared<NeuralMean>},
}))


DEFINE_STRINGABLE_ENUM(CovarianceFunctionFactory, "covariance", ({
    {"squared_exponential", std::make_shared<SquaredExponentialCovariance>},
    {"rational_quadratic", std::make_shared<RationalQuadraticCovariance>},
    {"matern", std::make_shared<MaternCovariance>},
    {"periodic", std::make_shared<PeriodicCovariance>},
    {"nonstationary", std::make_shared<NonstationaryCovariance>},
    {"mg-nonstationary", std::make_shared<MeanGradNonstationaryCovariance>},
    {"thin_plate", std::make_shared<ThinPlateCovariance>},
    {"neural", std::make_shared<NeuralNonstationaryCovariance>},
    {"proc_nonstationary", std::make_shared<ProceduralNonstationaryCovariance>},
    {"dot_product", std::make_shared<DotProductCovariance>},
}))


DEFINE_STRINGABLE_ENUM(ProceduralScalarFactory, "procedural_scalar", ({
    {"sdf", std::make_shared<ProceduralSdf>},
    {"noise", std::make_shared<ProceduralNoise>},
    {"regular_grid", std::make_shared<RegularGridScalar>},
}))

DEFINE_STRINGABLE_ENUM(ProceduralVectorFactory, "procedural_vector", ({
    {"regular_grid", std::make_shared<RegularGridVector>},
    {"noise", std::make_shared<ProceduralNoiseVec>},
}))

}
