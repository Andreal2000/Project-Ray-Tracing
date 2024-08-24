#pragma once

#include "vec3.h"
#include "color.h"

class Light
{
public:
    Point3 position;
    Color diffuse_color;
    Color specular_color;

    __device__ Light(const Point3 &position, const Color &diffuse_color, const Color &specular_color)
        : position(position), diffuse_color(diffuse_color), specular_color(specular_color) {}

    __device__ Light(const Point3 &position, float intensity)
        : position(position), diffuse_color(intensity * Color(1, 1, 1)), specular_color(intensity * Color(1, 1, 1)) {}

    __device__ Light(const Point3 &position, float diffuse_intensity, float specular_intensity)
        : position(position), diffuse_color(diffuse_intensity * Color(1, 1, 1)), specular_color(specular_intensity * Color(1, 1, 1)) {}

private:
};
