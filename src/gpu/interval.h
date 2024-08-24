#pragma once

#include <cmath>

class Interval
{
public:
    float min, max;

    __device__ Interval() : min(+INFINITY), max(-INFINITY) {} // Default interval is empty

    __device__ Interval(float min, float max) : min(min), max(max) {}

    __device__ Interval(const Interval &a, const Interval &b)
    {
        // Create the interval tightly enclosing the two input intervals.
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __device__ float size() const
    {
        return max - min;
    }

    __device__ bool contains(float x) const
    {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(float x) const
    {
        return min < x && x < max;
    }

    __device__ float clamp(float x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    __device__ Interval expand(float delta) const
    {
        auto padding = delta / 2;
        return Interval(min - padding, max + padding);
    }

    static const Interval empty, universe;
};

const Interval Interval::empty = Interval(+INFINITY, -INFINITY);
const Interval Interval::universe = Interval(-INFINITY, +INFINITY);