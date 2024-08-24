#pragma once

#include "hittable.h"

#define _USE_MATH_DEFINES
#include <math.h>

class Sphere : public Hittable
{
public:
    __device__ Sphere(Point3 &center, float radius, Texture *tex)
        : center(center), radius(radius), tex(tex)
    {
        auto rvec = Vec3(radius, radius, radius);
        bbox = AABB(center - rvec, center + rvec);
    };
    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override
    {
        Vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root))
        {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.tex = tex;

        return true;
    }

    __device__ AABB bounding_box() const override { return bbox; }

private:
    Point3 center;
    float radius;
    Texture *tex;
    AABB bbox;

    __device__ static void get_sphere_uv(const Point3 &p, float &u, float &v)
    {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acosf(-p.y());
        auto phi = atan2f(-p.z(), p.x()) + M_PI;

        u = phi / (2 * M_PI);
        v = theta / M_PI;
    }
};