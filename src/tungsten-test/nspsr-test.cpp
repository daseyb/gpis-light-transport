#include <math/GPNeuralNetwork.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <thread/ThreadUtils.hpp>
#include <math/GPFunctions.hpp>
#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif


using namespace Tungsten;

void nspsr_to_vdb(const GPNeuralNetwork& nn, int dim, Path base_path, std::vector<Vec3d> covTestP) {
    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    Vec3d minp = nn.bounds().min();
    Vec3d maxp = nn.bounds().max();

    auto nncov = NeuralNonstationaryCovariance(std::make_shared<GPNeuralNetwork>(nn));

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

    auto lsGrid = openvdb::createGrid<openvdb::Vec3dGrid>({ 1., 1., 1. });
    lsGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    lsGrid->setName("ls");
    lsGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::Vec3dGrid::Accessor lsAccessor = lsGrid->getAccessor();

    std::vector<std::shared_ptr<openvdb::FloatGrid>> cov2pGrids;
    std::vector<openvdb::FloatGrid::Accessor> cov2pAccessors;

    for (int i = 0; i < covTestP.size(); i++) {
        auto cov2pGrid = openvdb::createGrid<openvdb::FloatGrid>(1.f);
        cov2pGrid->setGridClass(openvdb::GRID_LEVEL_SET);
        cov2pGrid->setName(std::string("cov2p_") + std::to_string(i));
        cov2pGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
        cov2pGrids.push_back(cov2pGrid);
        cov2pAccessors.push_back(cov2pGrid->getAccessor());
    }

    for (int i = 0; i < dim; i++) {
        std::cout << i << "\r";
        for (int j = 0; j < dim; j++) {
#pragma omp parallel for
            for (int k = 0; k < dim; k++) {
                auto p = lerp(minp, maxp, Vec3d((float)i, (float)j, (float)k) / (dim));

                double mean = nn.mean(p);
                
                double sigma2 = nncov(Derivative::None, Derivative::None, p, p, Vec3d(0.), Vec3d(0.));

                auto ls_x = sqrt(sigma2 / nncov(Derivative::First, Derivative::First, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.)));
                auto ls_y = sqrt(sigma2 / nncov(Derivative::First, Derivative::First, p, p, Vec3d(0., 1., 0.), Vec3d(0., 1., 0.)));
                auto ls_z = sqrt(sigma2 / nncov(Derivative::First, Derivative::First, p, p, Vec3d(0., 0., 1.), Vec3d(0., 0., 1.)));

                auto local_cov = SquaredExponentialCovariance((float)sqrt(sigma2), 1.0f, Vec3f(1.) / Vec3f((float)(ls_x*ls_x), (float)(ls_y * ls_y), (float)(ls_z * ls_z)));

                std::vector<double> covTestVs(covTestP.size());
                for (int tidx = 0; tidx < covTestP.size(); tidx++) {
                    covTestVs[tidx] = local_cov(Derivative::None, Derivative::None, p, covTestP[tidx], Vec3d(0.), Vec3d(0.));
                }

                #pragma omp critical
                {
                    meanAccessor.setValue({ i,j,k }, mean);
                    varAccessor.setValue({ i,j,k }, sqrt(sigma2));
                    lsAccessor.setValue({ i,j,k }, { ls_x, ls_y, ls_z });
                    for (int tidx = 0; tidx < covTestVs.size(); tidx++) {
                        cov2pAccessors[tidx].setValue({ i,j,k }, covTestVs[tidx]);
                    }
                }
            }
        }
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        grids.push_back(varGrid);
        grids.push_back(lsGrid);
        for (auto cg : cov2pGrids) {
            grids.push_back(cg);
        }
        openvdb::io::File file(base_path.stripExtension().asString() + "-" + std::to_string(dim) + ".vdb");
        file.write(grids);
        file.close();
    }
}

int main() {

    Path file("example-scenes/gp-nn/bunny-network.json");

    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(file);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
    }

    GPNeuralNetwork network;
    network.read(*document, file.parent());

    std::cout << network.mean(Vec3d(0.1, 0.5, 0.1)) << "\n";
    std::cout << sqrt(network.cov(Vec3d(0.1, 0.5, 0.1), Vec3d(0.1, 0.5, 0.1))) << "\n";

    nspsr_to_vdb(network, 16, file, { Vec3d(0.3, 0.3, 0.3), Vec3d(-0.2, 0.3, -.2) });
}