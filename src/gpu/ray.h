#pragma once

#include "vec3.h"

class Ray
{
public:
  __device__ Ray() {}

  __device__ Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

  __device__ const Point3 &origin() const { return orig; }
  __device__ const Vec3 &direction() const { return dir; }

  __device__ Point3 at(float t) const { return orig + t * dir; }

private:
  Point3 orig;
  Vec3 dir;
};