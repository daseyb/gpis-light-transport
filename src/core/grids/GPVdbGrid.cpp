#include "GPVdbGrid.hpp"

#if OPENVDB_AVAILABLE

#include "sampling/PathSampleGenerator.hpp"

#include "math/BitManip.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

#include "Debug.hpp"

#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/MultiResGrid.h>
namespace Tungsten {

GPVdbGrid::GPVdbGrid()
:  _normalizeSize(true),
  _requestGradient(false)
{
}

void GPVdbGrid::fromJson(JsonPtr value, const Scene &scene)
{
    if (auto path = value["file"]) _path = scene.fetchResource(path);
    value.getField("normalize_size", _normalizeSize);
    value.getField("transform", _configTransform);
    value.getField("request_gradient", _requestGradient);

}

rapidjson::Value GPVdbGrid::toJson(Allocator &allocator) const
{
    JsonObject result{Grid::toJson(allocator), allocator,
        "type", "gp_vdb",
        "file", *_path,
        "normalize_size", _normalizeSize,
        "transform", _configTransform,
        "request_gradient", _requestGradient
    };

    return result;
}

#if 0
void GPVdbGrid::loadResources()
{
    openvdb::io::File file(_path->absolute().asString());
    try {
        file.open();
    } catch(const openvdb::IoError &e) {
        FAIL("Failed to open vdb file at '%s': %s", *_path, e.what());
    }

    openvdb::GridBase::Ptr ptr;
    try {
        ptr = file.readGrid(_densityName);
    } catch(const std::exception &) {
        ptr = nullptr;
    };
    if (!ptr)
        FAIL("Failed to read density grid '%s' from vdb file '%s'", _densityName, *_path);


    file.close();

    _densityGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(ptr);
    if (!_densityGrid)
        FAIL("Failed to read grid '%s' from vdb file '%s': Grid is not a FloatGrid", _densityName, *_path);

    auto accessor = _densityGrid->getAccessor();
    for (openvdb::FloatGrid::ValueOnIter iter = _densityGrid->beginValueOn(); iter.test(); ++iter)
        iter.setValue((*iter)*_densityScale);

    Vec3d densityCenter (ptr->transform().indexToWorld(openvdb::Vec3d(0, 0, 0)).asPointer());
    Vec3d densitySpacing(ptr->transform().indexToWorld(openvdb::Vec3d(1, 1, 1)).asPointer());
    densitySpacing -= densityCenter;

    Vec3d emissionCenter, emissionSpacing;
    if (emissionPtr) {
        emissionCenter  = Vec3d(emissionPtr->transform().indexToWorld(openvdb::Vec3d(0, 0, 0)).asPointer());
        emissionSpacing = Vec3d(emissionPtr->transform().indexToWorld(openvdb::Vec3d(1, 1, 1)).asPointer());
        emissionSpacing -= emissionCenter;
        _emissionGrid = openvdb::gridPtrCast<openvdb::Vec3fGrid>(emissionPtr);
    } else {
        emissionCenter = densityCenter;
        emissionSpacing = densitySpacing;
        _emissionGrid = nullptr;
    }
    _emissionIndexOffset = Vec3f((densityCenter - emissionCenter)/emissionSpacing);

    openvdb::CoordBBox bbox = _densityGrid->evalActiveVoxelBoundingBox();
    Vec3i minP = Vec3i(bbox.min().x(), bbox.min().y(), bbox.min().z());
    Vec3i maxP = Vec3i(bbox.max().x(), bbox.max().y(), bbox.max().z()) + 1;
    Vec3f diag = Vec3f(maxP - minP);

    float scale;
    Vec3f center;
    if (_normalizeSize) {
        scale = 1.0f/diag.max();
        diag *= scale;
        center = Vec3f(minP)*scale + Vec3f(diag.x(), 0.0f, diag.z())*0.5f;
    } else {
        scale = densitySpacing.min();
        center = -Vec3f(densityCenter);
    }

    _transform = Mat4f::translate(-center)*Mat4f::scale(Vec3f(scale));
    _invTransform = Mat4f::scale(Vec3f(1.0f/scale))*Mat4f::translate(center);
    _bounds = Box3f(Vec3f(minP), Vec3f(maxP));

    _invConfigTransform = _configTransform.invert();


    if (_densityName != "density" && _requestSDF && _densityGrid) {
        std::cout << "Converting density grid to SDF...\n";
        /*auto sdfGrid = openvdb::tools::fogToSdf(*_densityGrid, 0.0f);
        for (openvdb::FloatGrid::ValueOnIter iter = sdfGrid->beginValueOn(); iter.test(); ++iter) {
            iter.setValue((*iter) * -1);
        }*/
        
        int downsample = 2;
        openvdb::tools::MultiResGrid<openvdb::FloatTree> mgrid(downsample+1, _densityGrid);
       

        auto dilatedSdfGrid = openvdb::tools::dilateSdf(*mgrid.grid(downsample), 100, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, 20);

        //dilatedSdfGrid = openvdb::tools::sdfToSdf(*dilatedSdfGrid, 0, 10);

        float scaleFac = (1 << downsample);

        _transform = Mat4f::translate(-center) * Mat4f::scale(Vec3f(scale * scaleFac));
        _invTransform = Mat4f::scale(Vec3f(1.0f / (scale * scaleFac))) * Mat4f::translate(center);

        _densityGrid = dilatedSdfGrid;
    }

}


void GPVdbGrid::requestSDF() {
    _requestSDF = true;
}

Mat4f GPVdbGrid::naturalTransform() const
{
    return _configTransform*_transform;
}

Mat4f GPVdbGrid::invNaturalTransform() const
{
    return _invTransform*_invConfigTransform;
}

Box3f GPVdbGrid::bounds() const
{
    return _bounds;
}

template<typename TreeT>
static inline float gridAt(TreeT &acc, Vec3f p)
{
    return openvdb::tools::BoxSampler::sample(acc, openvdb::Vec3R(p.x(), p.y(), p.z()));
}

float GPVdbGrid::density(Vec3f p) const
{
    return gridAt(_densityGrid->tree(), p);
}

Vec3f GPVdbGrid::gradient(Vec3f p) const
{
    if (_gradientGrid) {
        return Vec3f(openvdb::tools::BoxSampler::sample(_gradientGrid->tree(), openvdb::Vec3R(p.x(), p.y(), p.z())).asPointer());
    }
    else {
        std::cerr << "Tried to evaluate gradient on grid for which it was not requested at initialization!" << std::endl;
        return Vec3f(0.f);
    }
}

#endif

}

#endif
