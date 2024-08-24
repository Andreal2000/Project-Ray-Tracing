#pragma once

#include "hittable.h"

class Quad : public Hittable
{
public:
    __device__ Quad(const Point3 &Q, const Vec3 &u, const Vec3 &v, Texture *tex)
        : Q(Q), u(u), v(v), tex(tex)
    {
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);

        set_bounding_box();
    }

    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override
    {
        auto denom = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (fabs(denom) < 1e-8)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        auto t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.contains(t))
            return false;

        // Determine the hit point lies within the planar shape using its plane coordinates.
        auto intersection = r.at(t);
        Vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t;
        rec.p = intersection;
        rec.tex = tex;
        rec.set_face_normal(r, normal);

        return true;
    }

    __device__ AABB bounding_box() const override { return bbox; }

private:
    Point3 Q;
    Vec3 u, v;
    Vec3 w;
    Texture *tex;
    Vec3 normal;
    float D;
    AABB bbox;

    __device__ bool is_interior(float a, float b, HitRecord &rec) const
    {
        Interval unit_interval = Interval(0, 1);
        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

    __device__ void set_bounding_box()
    {
        // Compute the bounding box of all four vertices.
        auto bbox_diagonal1 = AABB(Q, Q + u + v);
        auto bbox_diagonal2 = AABB(Q + u, Q + v);
        bbox = AABB(bbox_diagonal1, bbox_diagonal2);
    }
};