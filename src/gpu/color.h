#pragma once

#include "vec3.h"
#include "interval.h"

#include <iostream>

using Color = Vec3;

__device__ inline float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}   

__device__ void color_to_byte(const Color &pixel_color, uint8_t *rgb)
{
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0,1] component values to the byte range [0,255].
    const Interval intensity(0.000, 0.999);
    rgb[0] = static_cast<uint8_t>(256 * intensity.clamp(r));
    rgb[1] = static_cast<uint8_t>(256 * intensity.clamp(g));
    rgb[2] = static_cast<uint8_t>(256 * intensity.clamp(b));
}