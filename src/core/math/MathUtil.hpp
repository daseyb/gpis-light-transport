#ifndef MATHUTIL_HPP_
#define MATHUTIL_HPP_

#include "Vec.hpp"

#include "IntTypes.hpp"

#include <type_traits>
#include <utility>

namespace Tungsten {

template<typename T>
T min(const T &a, const T &b)
{
    return a < b ? a : b;
}

template<typename T, typename... Ts>
T min(const T &a, const T &b, const Ts &... ts)
{
    return min(min(a, b), ts...);
}

template<typename T>
T max(const T &a, const T &b)
{
    return a > b ? a : b;
}

template<typename T, typename... Ts>
T max(const T &a, const T &b, const Ts &... ts)
{
    return max(max(a, b), ts...);
}

template<typename ElementType, unsigned Size>
Vec<ElementType, Size> min(const Vec<ElementType, Size> &a, const Vec<ElementType, Size> &b)
{
    Vec<ElementType, Size> result(a);
    for (unsigned i = 0; i < Size; ++i)
        if (b.data()[i] < a.data()[i])
            result.data()[i] = b.data()[i];
    return result;
}

template<typename ElementType, unsigned Size>
Vec<ElementType, Size> max(const Vec<ElementType, Size> &a, const Vec<ElementType, Size> &b)
{
    Vec<ElementType, Size> result(a);
    for (unsigned i = 0; i < Size; ++i)
        if (b.data()[i] > a.data()[i])
            result.data()[i] = b.data()[i];
    return result;
}

template<typename T>
T abs(const T& a)
{
    return std::abs(a);
}

template<typename ElementType, unsigned Size>
Vec<ElementType, Size> abs(const Vec<ElementType, Size>& a)
{
    Vec<ElementType, Size> result(a);
    for (unsigned i = 0; i < Size; ++i)
        result.data()[i] = abs(a.data()[i]);
    return result;
}

template<typename T>
T clamp(T val, T minVal, T maxVal)
{
    return min(max(val, minVal), maxVal);
}

template<typename T>
T sqr(T val)
{
    return val*val;
}

template<typename T>
T cube(T val)
{
    return val*val*val;
}

template<typename T>
T lerp(T a, T b, T ratio)
{
    return a*(T(1) - ratio) + b*ratio;
}

static inline int intLerp(int x0, int x1, int t, int range)
{
    return (x0*(range - t) + x1*t)/range;
}

template<typename ElementType, unsigned Size>
Vec<ElementType, Size> lerp(const Vec<ElementType, Size> &a, const Vec<ElementType, Size> &b, ElementType ratio)
{
    return a*(ElementType(1) - ratio) + b*ratio;
}

template<typename T>
T smoothStep(T edge0, T edge1, T x) {
    x = clamp((x - edge0)/(edge1 - edge0), T(0), T(1));

    return x*x*(T(3) - T(2)*x);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static inline float trigInverse(float x)
{
    return min(std::sqrt(max(1.0f - x*x, 0.0f)), 1.0f);
}

static inline float trigDoubleAngle(float x)
{
    return clamp(2.0f*x*x - 1.0f, -1.0f, 1.0f);
}

static inline float trigHalfAngle(float x)
{
    return min(std::sqrt(max(x*0.5f + 0.5f, 0.0f)), 1.0f);
}


template <typename T, typename = int>
struct HasXMem : std::false_type { };

template <typename T>
struct HasXMem <T, decltype(std::declval<T>().x(), 0)> : std::true_type { };

template<typename To, typename From>
inline To vec_conv(const From& vd) {
    if constexpr (HasXMem<From>::value) {
        if constexpr (HasXMem<To>::value) {
            using ToElemType = typename std::remove_reference<decltype(std::declval<To>().x())>::type;
            return To{ (ToElemType)vd.x(), (ToElemType)vd.y(), (ToElemType)vd.z() };
        }
        else {
            using ToElemType = typename std::remove_reference<decltype(std::declval<To>().x)>::type;
            return To{ (ToElemType)vd.x(), (ToElemType)vd.y(), (ToElemType)vd.z() };
        }
    }
    else {
        if constexpr (HasXMem<To>::value) {
            using ToElemType = typename std::remove_reference<decltype(std::declval<To>().x())>::type;
            return To{ (ToElemType)vd.x, (ToElemType)vd.y, (ToElemType)vd.z };
        }
        else {
            using ToElemType = typename std::remove_reference<decltype(std::declval<To>().x)>::type;
            return To{ (ToElemType)vd.x, (ToElemType)vd.y, (ToElemType)vd.z };
        }
    }
}

// TODO: Review which of these are still in use
class MathUtil
{
public:
    // TODO: Is this a good hash? Try to track down the source of this
    static inline uint32 hash32(uint32 x) {
        x = ~x + (x << 15);
        x = x ^ (x >> 12);
        x = x + (x << 2);
        x = x ^ (x >> 4);
        x = x * 2057;
        x = x ^ (x >> 16);
        return x;
    }

    static float sphericalDistance(float lat0, float long0, float lat1, float long1, float r)
    {
        float  latSin = std::sin(( lat1 -  lat0)*0.5f);
        float longSin = std::sin((long1 - long0)*0.5f);
        return 2.0f*r*std::asin(std::sqrt(latSin*latSin + std::cos(lat0)*std::cos(lat1)*longSin*longSin));
    }

    // See http://geomalgorithms.com/a07-_distance.html
    static Vec2f closestPointBetweenLines(const Vec3f& P0, const Vec3f& u, const Vec3f& Q0, const Vec3f& v)
    {
        const Vec3f w0 = P0 - Q0;
        const float a = u.dot(u);
        const float b = u.dot(v);
        const float c = v.dot(v);
        const float d = u.dot(w0);
        const float e = v.dot(w0);
        float denom = a*c - b*b;
        if (denom == 0.0f)
            return Vec2f(0.0f);
        else
            return Vec2f(b*e - c*d, a*e - b*d)/denom;
    }

    static float triangleArea(const Vec3f &a, const Vec3f &b, const Vec3f &c)
    {
        return (b - a).cross(c - a).length()*0.5f;
    }
};

static float blackbody_table_r[][3] = {
 {1.61919106e+03f, -2.05010916e-03f, 5.02995757e+00f},
 {2.48845471e+03f, -1.11330907e-03f, 3.22621544e+00f},
 {3.34143193e+03f, -4.86551192e-04f, 1.76486769e+00f},
 {4.09461742e+03f, -1.27446582e-04f, 7.25731635e-01f},
 {4.67028036e+03f, 2.91258199e-05f, 1.26703442e-01f},
 {4.59509185e+03f, 2.87495649e-05f, 1.50345020e-01f},
 {3.78717450e+03f, 9.35907826e-06f, 3.99075871e-01f}
};

static float blackbody_table_g[][3] = {
 {-4.88999748e+02f, 6.04330754e-04f, -7.55807526e-02f},
 {-7.55994277e+02f, 3.16730098e-04f, 4.78306139e-01f},
 {-1.02363977e+03f, 1.20223470e-04f, 9.36662319e-01f},
 {-1.26571316e+03f, 4.87340896e-06f, 1.27054498e+00f},
 {-1.42529332e+03f, -4.01150431e-05f, 1.43972784e+00f},
 {-1.17554822e+03f, -2.16378048e-05f, 1.30408023e+00f},
 {-5.00799571e+02f, -4.59832026e-06f, 1.09098763e+00f}
};

static float blackbody_table_b[][4] = {
 {5.96945309e-11f, -4.85742887e-08f, -9.70622247e-05f, -4.07936148e-03f},
 {2.40430366e-11f, 5.55021075e-08f, -1.98503712e-04f, 2.89312858e-02f},
 {-1.40949732e-11f, 1.89878968e-07f, -3.56632824e-04f, 9.10767778e-02f},
 {-3.61460868e-11f, 2.84822009e-07f, -4.93211319e-04f, 1.56723440e-01f},
 {-1.97075738e-11f, 1.75359352e-07f, -2.50542825e-04f, -2.22783266e-02f},
 {-1.61997957e-13f, -1.64216008e-08f, 3.86216271e-04f, -7.38077418e-01f},
 {6.72650283e-13f, -2.73078809e-08f, 4.24098264e-04f, -7.52335691e-01f}
};

static inline Vec3f blackbody_color_rec709(float t)
{
  /* Calculate color in range 800..12000 using an approximation
   * a/x+bx+c for R and G and ((at + b)t + c)t + d) for B.
   *
   * The result of this can be negative to support gamut wider than
   * than rec.709, just needs to be clamped. */

  if (t >= 12000.0f) {
    return Vec3f(0.8262954810464208f, 0.9945080501520986f, 1.566307710274283f);
  }
  else if (t < 800.0f) {
    /* Arbitrary lower limit where light is very dim, matching OSL. */
    return Vec3f(5.413294490189271f, -0.20319390035873933f, -0.0822535242887164f);
  }

  int i = (t >= 6365.0f) ? 6 :
          (t >= 3315.0f) ? 5 :
          (t >= 1902.0f) ? 4 :
          (t >= 1449.0f) ? 3 :
          (t >= 1167.0f) ? 2 :
          (t >= 965.0f)  ? 1 :
                           0;

  const float *r = blackbody_table_r[i];
  const float *g = blackbody_table_g[i];
  const float *b = blackbody_table_b[i];

  const float t_inv = 1.0f / t;
  return Vec3f(r[0] * t_inv + r[1] * t + r[2],
                     g[0] * t_inv + g[1] * t + g[2],
                     ((b[0] * t + b[1]) * t + b[2]) * t + b[3]);
}

}

#endif /* MATHUTIL_HPP_ */
