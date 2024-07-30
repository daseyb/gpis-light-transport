#ifndef SDFFUNCTIONS_HPP_
#define SDFFUNCTIONS_HPP_

#include <math/Vec.hpp>

namespace Tungsten {
	double fbm(Vec3d uv, int octaves);

	class SdfFunctions {
	public:

		enum class Function {
			Knob,
			KnobInner,
			KnobOuter,
			TwoSpheres
		};

		static std::string functionToString(Function val);
		static Function stringToFunction(const std::string& name);

		static float knob(Vec3f p, int& mat);
		static float knob_inner(Vec3f p, int& mat);
		static float knob_outer(Vec3f p, int& mat);
		static float two_spheres(Vec3f p, int& mat);
		
		template<typename sdf>
		static Vec3f grad(sdf func, Vec3f p) {
			constexpr float eps = 0.001f;

			int mat;
			std::array<float, 4> vals = {
				func(p + Vec3f(eps, 0.f, 0.f), mat),
				func(p + Vec3f(0.f, eps, 0.f), mat),
				func(p + Vec3f(0.f, 0.f, eps), mat),
				func(p, mat)
			};

			return Vec3f(
				vals[0] - vals[3],
				vals[1] - vals[3],
				vals[2] - vals[3]
			) / eps;
		}

		static float eval(Function fn, Vec3f p, int& mat) {
			switch (fn) {
			case Function::Knob: return knob(p, mat);
			case Function::KnobInner: return knob_inner(p, mat);
			case Function::KnobOuter: return knob_outer(p, mat);
			case Function::TwoSpheres: return two_spheres(p, mat);
			}
		}

		static Vec3f grad(Function fn, Vec3f p) {
			switch (fn) {
			case Function::Knob: return grad(knob, p);
			case Function::KnobInner: return grad(knob_inner, p);
			case Function::KnobOuter: return grad(knob_outer, p);
			case Function::TwoSpheres: return grad(two_spheres, p);
			}
		}

	};

}

#endif //SDFFUNCTIONS_HPP_