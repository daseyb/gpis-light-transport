#include "LambertianPhaseFunction.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "samplerecords/MediumSample.hpp"

#include "io/JsonObject.hpp"


namespace Tungsten {

rapidjson::Value LambertianPhaseFunction::toJson(Allocator &allocator) const
{
    return JsonObject{PhaseFunction::toJson(allocator), allocator,
        "type", "lambertian"
    };
}

}
