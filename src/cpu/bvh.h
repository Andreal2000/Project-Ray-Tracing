#pragma once

#include "utils.h"
#include "aabb.h"
#include "hittable.h"

#include <algorithm>

class BVHNode : public Hittable
{
public:
    BVHNode(std::vector<shared_ptr<Hittable>> list) : BVHNode(list, 0, list.size())
    {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
    }

    BVHNode(std::vector<shared_ptr<Hittable>> &objects, size_t start, size_t end)
    {
        // Build the bounding box of the span of source objects.
        bbox = AABB::empty;
        for (size_t object_index = start; object_index < end; object_index++)
            bbox = AABB(bbox, objects[object_index]->bounding_box());

        int axis = bbox.longest_axis();

        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1)
        {
            left = right = objects[start];
        }
        else if (object_span == 2)
        {
            left = objects[start];
            right = objects[start + 1];
        }
        else
        {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            auto mid = start + object_span / 2;
            left = make_shared<BVHNode>(objects, start, mid);
            right = make_shared<BVHNode>(objects, mid, end);
        }
    }

    bool hit(const Ray &r, const Interval ray_t, HitRecord &rec) const override
    {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, Interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    AABB bounding_box() const override { return bbox; }

private:
    shared_ptr<Hittable> left;
    shared_ptr<Hittable> right;
    AABB bbox;

    static bool box_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b, int axis_index)
    {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    static bool box_x_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b)
    {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b)
    {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b)
    {
        return box_compare(a, b, 2);
    }
};