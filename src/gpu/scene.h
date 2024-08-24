#pragma once

#include "hittable.h"
#include "light.h"

class Scene : public Hittable
{
public:
    Hittable **objects;
    int objects_size;

    Light **lights;
    int lights_size;

    Color ambient_color;

    bool use_bounding_box = false;

    __device__ Scene()
        : ambient_color(Color(0, 0, 0)) { set_bounding_box(); }
    __device__ Scene(Hittable **l, int n)
        : objects(l), objects_size(n), ambient_color(Color(0, 0, 0)) { set_bounding_box(); }
    __device__ Scene(Hittable **l, int n, Light **lights, int lights_size)
        : objects(l), objects_size(n), lights(lights), lights_size(lights_size), ambient_color(Color(0, 0, 0)) { set_bounding_box(); }

    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const
    {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;
        for (int i = 0; i < objects_size; i++)
        {
            if (!use_bounding_box || objects[i]->bounding_box().hit(r, Interval(ray_t.min, closest_so_far)))
            {
                if (objects[i]->hit(r, Interval(ray_t.min, closest_so_far), temp_rec))
                {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
        }
        return hit_anything;
    }

    __device__ AABB bounding_box() const override { return bbox; }

private:
    AABB bbox;

    __device__ void set_bounding_box()
    {
        for (int i = 0; i < objects_size; i++)
        {
            bbox = AABB(bbox, objects[i]->bounding_box());
        }
    }
};