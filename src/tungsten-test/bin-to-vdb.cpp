#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <thread/ThreadUtils.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;


int gen3d(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto base_path = Path(argv[1]);

    std::vector<double> mean, var;
    std::vector<Vec3d> gridp;
    {
        std::ifstream xfile(base_path.asString() + "-mean.bin", std::ios::in | std::ios::binary);

        if(!xfile.is_open())
            std::cerr << "Huh\n";

        // Read the doubles from the file
        double value;
        while (xfile.read(reinterpret_cast<char*>(&value), sizeof(double))) {
            mean.push_back(value);
        }

        // Close the file
        xfile.close();
    }

    {
        std::ifstream xfile(base_path.asString() + "-var.bin", std::ios::in | std::ios::binary);

        // Read the doubles from the file
        double value;
        while (xfile.read(reinterpret_cast<char*>(&value), sizeof(double))) {
            var.push_back(value);
        }

        // Close the file
        xfile.close();
    }

    Vec3d minp = Vec3d(DBL_MAX);
    Vec3d maxp = Vec3d(-DBL_MAX);

    {
        std::ifstream xfile(base_path.asString() + "-grid_vertices.bin", std::ios::in | std::ios::binary);

        // Read the doubles from the file
        Vec3d value;
        while (xfile.read(reinterpret_cast<char*>(&value), sizeof(Vec3d))) {
            gridp.push_back(value);
            minp = min(minp, value);
            maxp = max(maxp, value);
        }

        // Close the file
        xfile.close();
    }

    size_t dim = cbrt(gridp.size());


    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>((maxp - minp) / dim));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    auto varGrid = openvdb::createGrid<openvdb::FloatGrid>(1.f);
    varGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    varGrid->setName("variance");
    varGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor varAccessor = varGrid->getAccessor();

    for (int i = 0; i < gridp.size(); i++) {
        auto localP = meanGrid->transform().worldToIndexNodeCentered(vec_conv<openvdb::Vec3d>(gridp[i]));
        meanAccessor.setValue({ localP.y(), localP.x(), localP.z() }, mean[i]);
        varAccessor.setValue(localP, sqrt(var[i]));
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        grids.push_back(varGrid);
        openvdb::io::File file(base_path.asString() + ".vdb");
        file.write(grids);
        file.close();
    }

    return 0;

}

template<typename GridType>
int grid_to_vdb(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto base_path = Path(argv[1]);

    auto grid = load_grid<GridType::ValueType>(base_path);


    Vec3d minp = grid.bounds.min();
    Vec3d maxp = grid.bounds.max();

    size_t dim = grid.res;

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>((maxp - minp) / dim));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<GridType>();
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("values");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    GridType::Accessor meanAccessor = meanGrid->getAccessor();

    auto points = grid.makePoints(true);
    
    for (int i = 0; i < points.size(); i++) {
        auto localP = meanGrid->transform().worldToIndexNodeCentered(vec_conv<openvdb::Vec3d>(points[i]));
        meanAccessor.setValue({ localP.y(), localP.x(), localP.z() }, grid.values[i]);
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        openvdb::io::File file(base_path.stripExtension().asString() + ".vdb");
        file.write(grids);
        file.close();
    }

    return 0;
}

int main(int argc, char** argv) {

    if (Path(argv[1]).extension() == ".json") {
        if (argc == 2 || std::string(argv[2]) == "1") {
            return grid_to_vdb<openvdb::DoubleGrid>(argc, argv);
        }
        else if(std::string(argv[2]) == "3") {
            return grid_to_vdb<openvdb::Vec3DGrid>(argc, argv);
        }
    }
    else {
        return gen3d(argc, argv);
    }
}
