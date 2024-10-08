#pragma once

#include "utils.h"

class AABB
{
public:
    Interval x, y, z;

    AABB() {} // The default AABB is empty, since intervals are empty by default.

    AABB(const Interval &x, const Interval &y, const Interval &z)
        : x(x), y(y), z(z)
    {
        pad_to_minimums();
    }

    AABB(const Point3 &a, const Point3 &b)
    {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.

        x = Interval(fmin(a[0], b[0]), fmax(a[0], b[0]));
        y = Interval(fmin(a[1], b[1]), fmax(a[1], b[1]));
        z = Interval(fmin(a[2], b[2]), fmax(a[2], b[2]));

        pad_to_minimums();
    }

    AABB(const AABB &box0, const AABB &box1)
    {
        x = Interval(box0.x, box1.x);
        y = Interval(box0.y, box1.y);
        z = Interval(box0.z, box1.z);
    }

    const Interval &axis_interval(int n) const
    {
        if (n == 1)
            return y;
        if (n == 2)
            return z;
        return x;
    }

    bool hit(const Ray &r, Interval ray_t) const
    {
        const Point3 &ray_orig = r.origin();
        const Vec3 &ray_dir = r.direction();

        for (int axis = 0; axis < 3; axis++)
        {
            const Interval &ax = axis_interval(axis);
            const float adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1)
            {
                if (t0 > ray_t.min)
                    ray_t.min = t0;
                if (t1 < ray_t.max)
                    ray_t.max = t1;
            }
            else
            {
                if (t1 > ray_t.min)
                    ray_t.min = t1;
                if (t0 < ray_t.max)
                    ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }

    int longest_axis() const
    {
        // Returns the index of the longest axis of the bounding box.

        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    static const AABB empty, universe;

private:
    void pad_to_minimums()
    {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.

        float delta = 0.001;
        if (x.size() < delta)
            x = x.expand(delta);
        if (y.size() < delta)
            y = y.expand(delta);
        if (z.size() < delta)
            z = z.expand(delta);
    }
};

const AABB AABB::empty = AABB(Interval::empty, Interval::empty, Interval::empty);
const AABB AABB::universe = AABB(Interval::universe, Interval::universe, Interval::universe);