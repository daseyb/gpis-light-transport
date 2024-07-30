#ifndef GAUSSIANPROCESSFACTORY_HPP_
#define GAUSSIANPROCESSFACTORY_HPP_

#include "StringableEnum.hpp"

#include <functional>
#include <memory>

namespace Tungsten {

class GPSampleNode;
class CovarianceFunction;
class MeanFunction;
class ProceduralScalar;
class ProceduralVector;

typedef StringableEnum<std::function<std::shared_ptr<GPSampleNode>()>> GaussianProcessFactory;
typedef StringableEnum<std::function<std::shared_ptr<CovarianceFunction>()>> CovarianceFunctionFactory;
typedef StringableEnum<std::function<std::shared_ptr<MeanFunction>()>> MeanFunctionFactory;
typedef StringableEnum<std::function<std::shared_ptr<ProceduralScalar>()>> ProceduralScalarFactory;
typedef StringableEnum<std::function<std::shared_ptr<ProceduralVector>()>> ProceduralVectorFactory;

}

#endif /* GAUSSIANPROCESSFACTORY_HPP_ */
