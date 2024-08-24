#pragma once

#include "aabb.h"
#include "ray.h"
#include "texture.h"

class HitRecord
{
public:
    Point3 p;
    Vec3 normal;
    shared_ptr<Texture> tex;
    float t;
    float u;
    float v;
    bool front_face;

    void set_face_normal(const Ray &r, const Vec3 &outward_normal)
    {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable
{
public:
    virtual ~Hittable() = default;

    virtual bool hit(const Ray &r, const Interval ray_t, HitRecord &rec) const = 0;

    virtual AABB bounding_box() const = 0;
};