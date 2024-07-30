#ifndef VERTEX_HPP_
#define VERTEX_HPP_

#include "math/Vec.hpp"

#include <type_traits>

namespace Tungsten {

class Vertex
{
    Vec3f _pos, _normal;
    Vec2f _uv;
    Vec3f _color;

public:
    Vertex() = default;

    Vertex(const Vec3f &pos)
    : _pos(pos)
    {
    }

    Vertex(const Vec3f &pos, const Vec2f &uv)
    : _pos(pos), _uv(uv)
    {
    }

    Vertex(const Vec3f &pos, const Vec3f &normal, const Vec2f &uv)
    : _pos(pos), _normal(normal), _uv(uv)
    {
    }

    Vertex(const Vec3f& pos, const Vec3f& normal, const Vec2f& uv, const Vec3f& color)
        : _pos(pos), _normal(normal), _uv(uv), _color(color)
    {
    }


    const Vec3f &normal() const
    {
        return _normal;
    }

    const Vec3f &pos() const
    {
        return _pos;
    }

    const Vec2f &uv() const
    {
        return _uv;
    }

    const Vec3f& color() const
    {
        return _color;
    }

    Vec3f &normal()
    {
        return _normal;
    }

    Vec3f &pos()
    {
        return _pos;
    }

    Vec2f &uv()
    {
        return _uv;
    }

    Vec3f& color()
    {
        return _color;
    }

    bool operator==(const Vertex& o) const
    {
        if (pos() != o.pos()) return false;
        if (uv() != o.uv()) return false;
        if (normal() != o.normal()) return false;
        if (color() != o.color()) return false;
        return true;
    }

    bool operator!=(const Vertex& o) const
    {
        if (pos() != o.pos()) return true;
        if (uv() != o.uv()) return true;
        if (normal() != o.normal()) return true;
        if (color() != o.color()) return true;
        return false;
    }

};

// MSVC's views on what is POD or not differ from gcc or clang.
// memcpy and similar code still seem to work, so we ignore this
// issue for now.
#ifndef _MSC_VER
static_assert(std::is_pod<Vertex>::value, "Vertex needs to be of POD type!");
#endif

}

template <class T>
inline void hash_combine(std::size_t& s, const T& v)
{
    std::hash<T> h;
    s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

namespace std
{
    template <>
    struct hash<Tungsten::Vertex>
    {
        std::size_t operator()(const Tungsten::Vertex& c) const
        {
            std::size_t result = 0;
            hash_combine(result, c.pos());
            hash_combine(result, c.normal());
            hash_combine(result, c.uv());
            hash_combine(result, c.color());
            return result;
        }
    };
}

#endif /* VERTEX_HPP_ */
