#pragma once

#include "utils.h"
#include "bvh.h"

class Scene : public Hittable
{
public:
    std::vector<shared_ptr<Hittable>> objects;
    std::vector<shared_ptr<Light>> lights;
    Color ambient_color;
    bool use_bounding_box = false;

    Scene() : ambient_color(Color(0, 0, 0)) {}

    void clear_all()
    {
        clear_objects();
        clear_lights();
    }

    void clear_objects() { objects.clear(); }
    void clear_lights() { lights.clear(); }

    void add_object(shared_ptr<Hittable> object)
    {
        objects.push_back(object);
        bbox = AABB(bbox, object->bounding_box());
    }
    void add_light(shared_ptr<Light> light) { lights.push_back(light); }

    void create_bvh()
    {
        std::shared_ptr<Hittable> bvh = make_shared<BVHNode>(objects);
        clear_objects();
        add_object(bvh);
    }

    bool hit(const Ray &r, const Interval ray_t, HitRecord &rec) const override
    {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto &object : objects)
        {
            if (!use_bounding_box || object->bounding_box().hit(r, Interval(ray_t.min, closest_so_far)))
            {
                if (object->hit(r, Interval(ray_t.min, closest_so_far), temp_rec))
                {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
        }

        return hit_anything;
    }

    AABB bounding_box() const override { return bbox; }

private:
    AABB bbox;
};