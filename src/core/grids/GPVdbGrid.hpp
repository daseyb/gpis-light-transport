#ifndef GPVDBGRID_HPP_
#define GPVDBGRID_HPP_

#if OPENVDB_AVAILABLE

#include "Grid.hpp"

#include "io/FileUtils.hpp"

#include <openvdb/openvdb.h>

#include <Eigen/Dense>

namespace Tungsten {

class GPVdbGrid : public Grid
{
    PathPtr _path;
    bool _normalizeSize;
    Mat4f _configTransform;
    Mat4f _invConfigTransform;

    openvdb::FloatGrid::Ptr _meanGrid;
    openvdb::FloatGrid::Ptr _varianceGrid;
    std::vector< openvdb::FloatGrid::Ptr> _anisoGrids;

    Mat4f _transform;
    Mat4f _invTransform;
    Box3f _bounds;
    bool _requestGradient;
    bool _requestSDF;
    

public:
    GPVdbGrid();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual void loadResources() override;

    virtual Mat4f naturalTransform() const override;
    virtual Mat4f invNaturalTransform() const override;
    virtual Box3f bounds() const override;

    virtual void requestGradient() override;
    virtual void requestSDF() override;

    float mean(Vec3f p);
    float variance(Vec3f p);
    Eigen::Matrix3f ansio(Vec3f p);

    Vec3f gradient(Vec3f p) const override;
};

}

#endif

#endif /* GPVDBGRID_HPP_ */
