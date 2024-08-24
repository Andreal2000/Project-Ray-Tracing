#pragma once

#include "aabb.h"
#include "ray.h"
#include "interval.h"
#include "texture.h"

struct HitRecord
{
    Point3 p;
    Vec3 normal;
    Texture *tex;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ void set_face_normal(const Ray &r, const Vec3 &outward_normal)
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
    __device__ virtual ~Hittable(){};
    __device__ virtual bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const = 0;
    __device__ virtual AABB bounding_box() const = 0;
};
