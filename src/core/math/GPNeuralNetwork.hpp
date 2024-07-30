#pragma once

#include <Eigen/Dense>
#include <math/MathUtil.hpp>
#include <math/Vec.hpp>
#include <math/Box.hpp>
#include <vector>
#include <fstream>
#include <io/JsonDocument.hpp>
#include <io/Scene.hpp>
#include <io/JsonSerializable.hpp>

namespace Tungsten {

template<typename T>
T softplus(const T& in, int beta = 1, int threshold = 20) {
	if (in * beta > threshold)
		return in;
	return T(1.0) / beta * log(T(1.0) + exp(beta * in));
}

class GPNeuralNetwork : public JsonSerializable {

	struct LinearLayer {
		Eigen::VectorXd biases;
		Eigen::MatrixXd weights;

		void read(std::ifstream& file, size_t in_features, size_t out_features) {
			weights = Eigen::MatrixXd(out_features, in_features);
			file.read((char*)weights.data(), sizeof(weights[0]) * weights.rows() * weights.cols());

			biases = Eigen::VectorXd(out_features);
			file.read((char*)biases.data(), sizeof(biases[0]) * biases.size());
		}

		template<typename VecT>
		VecT infer(const VecT& in) const {
			return weights * in + biases;
		}
	};

	struct MLP {
		LinearLayer inputLayer, outputLayer;
		std::vector<LinearLayer> layers;

		void read(std::ifstream& file, JsonPtr layerDescs) {
			
			if (!layerDescs.isArray()) {
				std::cerr << "Layers should be an array\n";
			}

			int i = 0;
			inputLayer.read(file, layerDescs[i][0u].cast<int>(), layerDescs[i][1u].cast<int>());

			for (i = 1; i < layerDescs.size() - 1; i++) {
				LinearLayer layer;
				layer.read(file, layerDescs[i][0u].cast<int>(), layerDescs[i][1u].cast<int>());
				layers.push_back(layer);
			}
			
			outputLayer.read(file, layerDescs[i][0u].cast<int>(), (int)layerDescs[i][1u].cast<int>());
		}
		
		//template<typename VecT>
		//VecT infer(const VecT& in) const {
		//	VecT v = inputLayer.infer(in);
		//	for (size_t i = 0; i < layers.size(); i++) {
		//		v = layers[i].infer(v);
		//		v = v.array().sin().matrix();
		//	}
		//	return outputLayer.infer(v);
		//}

		template<typename Scalar>
		Eigen::Vector<Scalar,-1> infer(const Eigen::Vector<Scalar, -1>& in) const {
			Eigen::Vector<Scalar, -1> v = inputLayer.infer(in);
			for (size_t i = 0; i < layers.size(); i++) {
				v = layers[i].infer(v);
				v = v.array().sin().matrix();
			}
			return outputLayer.infer(v);
		}
	};


	struct CovNetwork {

		MLP mlp;

		void read(std::ifstream& file, JsonPtr layerDescs) {
			mlp.read(file, layerDescs);
		}

		template<typename Scalar>
		Scalar infer(const Eigen::Vector<Scalar, 3>& a, const Eigen::Vector<Scalar, 3>& b) const {
			Eigen::Vector<Scalar, 6> inputAB = Eigen::Vector<Scalar, 6>({ {
				a.x(), a.y(), a.z(), b.x(), b.y(), b.z()
			} });

			Eigen::Vector<Scalar, 6> inputBA = Eigen::Vector<Scalar, 6>({ {
				b.x(), b.y(), b.z(), a.x(), a.y(), a.z()
			} });

			return softplus(mlp.infer<Scalar>(inputAB)(0)) + softplus(mlp.infer<Scalar>(inputBA)(0));
		}

	};

	MLP meanNetwork;
	CovNetwork covNetwork;
	Box3d _bounds;

public:

	GPNeuralNetwork() {

	}

	void fromJson(JsonPtr value, const Scene& scene) {
		JsonSerializable::fromJson(value, scene);
		read(value, scene.path().parent());
	}


	void read(JsonPtr desc, Path env) {

		Vec3d bmin, bmax;
		desc["bounds"].getField("min", bmin);
		desc["bounds"].getField("max", bmax);
		_bounds = Box3d(bmin, bmax);

		{
			std::string meanFile = env.asString() + "/" + desc["mean"]["file"].cast<std::string>();
			std::ifstream file(meanFile, std::ios::in | std::ios::binary);
			if (!file.is_open()) {
				std::cerr << "Couldn't open mean file " << meanFile << "\n";
			}
			meanNetwork.read(file, desc["mean"]["layers"]);
		}

		{
			std::string covFile = env.asString() + "/" + desc["cov"]["file"].cast<std::string>();
			std::ifstream file(covFile, std::ios::in | std::ios::binary);
			if (!file.is_open()) {
				std::cerr << "Couldn't open cov file " << covFile << "\n";
			}

			covNetwork.read(file, desc["cov"]["layers"]);
		}
	}

	template<typename Scalar>
	Scalar mean(Eigen::Vector<Scalar, 3>& p) const {
		if (!_bounds.contains(vec_conv<Vec3d>(p))) {
			p = p.cwiseMin(vec_conv<Eigen::Vector<Scalar, 3>>(_bounds.max())).cwiseMax(vec_conv<Eigen::Vector<Scalar, 3>>(_bounds.min()));
		}
		return meanNetwork.infer<Scalar>(p)(0);
	}
	
	double mean(Vec3d p) const {
		auto pv =vec_conv<Eigen::Vector3d>(p);
		return mean(pv);
	}

	double cov(Vec3d a, Vec3d b) const {
		auto av = vec_conv<Eigen::Vector3d>(a);
		auto bv = vec_conv<Eigen::Vector3d>(b);
		return cov(av, bv);
	}

	template<typename Scalar>
	Scalar cov(const Eigen::Vector<Scalar, 3>& a, const Eigen::Vector<Scalar, 3>& b) const {
		if (!(_bounds.contains(vec_conv<Vec3d>(a)) && _bounds.contains(vec_conv<Vec3d>(b)))) {
			return 0.000001;
		}
		return covNetwork.infer(a,b);
	}

	Box3d bounds() const {
		return _bounds;
	}
};

}