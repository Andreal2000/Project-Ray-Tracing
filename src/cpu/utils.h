#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <random>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Utility Functions

inline float degrees_to_radians(float degrees)
{
    return degrees * M_PI / 180.0;
}

inline float random_float()
{
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float random_float(float min, float max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

inline int random_int(int min, int max)
{
    // Returns a random integer in [min,max].
    return int(random_float(min, max + 1));
}

// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"
#include "light.h"

// Macro defines

#define MEASURE_DURATION(name, code_block)                                               \
{                                                                                        \
    std::clog << name;                                                                   \
    auto start = std::chrono::high_resolution_clock::now();                              \
    code_block                                                                           \
    auto stop = std::chrono::high_resolution_clock::now();                               \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); \
    std::clog << " took " << duration.count() << "ms" << std::endl;                      \
}