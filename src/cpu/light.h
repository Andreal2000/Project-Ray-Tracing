#pragma once

#include "vec3.h"
#include "color.h"

class Light
{
public:
    Point3 position;
    Color diffuse_color;
    Color specular_color;

    Light(const Point3 &position, const Color &diffuse_color, const Color &specular_color)
        : position(position), diffuse_color(diffuse_color), specular_color(specular_color) {}

    template <typename T>
    Light(const Point3 &position, T intensity)
        : position(position), diffuse_color(static_cast<float>(intensity) * Color(1, 1, 1)), specular_color(static_cast<float>(intensity) * Color(1, 1, 1)) {}

    template <typename T>
    Light(const Point3 &position, T diffuse_intensity, T specular_intensity)
        : position(position), diffuse_color(static_cast<float>(diffuse_intensity) * Color(1, 1, 1)), specular_color(static_cast<float>(specular_intensity) * Color(1, 1, 1)) {}

private:
};
