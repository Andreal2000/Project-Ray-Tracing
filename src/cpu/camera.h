#pragma once

#include "utils.h"
#include "scene.h"
#include "output_image.h"
#include "gif.h"

#include <fstream>
#include <chrono>
#include <vector>

class Camera
{
public:
    bool antialiasing = false; // Enable or disable antialiasing
    float aspect_ratio = 1.0;  // Ratio of image width over height
    int image_width = 100;     // Rendered image width in pixel count
    int samples_per_pixel = 8; // Count of random samples for each pixel

    float vfov = 90;                   // Vertical view angle (field of view)
    Point3 lookfrom = Point3(0, 0, 0); // Point camera is looking from
    Point3 lookat = Point3(0, 0, 1);   // Point camera is looking at
    Vec3 vup = Vec3(0, 1, 0);          // Camera-relative "up" direction

    void render(const Scene &world, OutputImage &output)
    {
        std::vector<Color> buffer;

        MEASURE_DURATION("Rendering",
                         initialize();

                         buffer = render_pixels(world););

        MEASURE_DURATION("Exporting", output.write_image(buffer, image_width, image_height););
    }

    void render_gif(const Scene &world, const std::string &filename, int duration, int fps, int quality, bool loop)
    {
        assert(quality >= 2 && quality <= 8);

        int frames_size = (duration * 0.001) * fps;

        Vec3 original_lookfrom = lookfrom;

        std::vector<std::vector<Color>> frames;

        MEASURE_DURATION("Rendering",
                         {
                             for (int f = 0; f < frames_size; f++)
                             {
                                 lookfrom = original_lookfrom;

                                 rotate_around_look_at((360.0 / frames_size) * f);

                                 initialize();

                                 frames.push_back(render_pixels(world));
                             }
                         });
        MEASURE_DURATION("Exporting",
                         Gif gif(frames, image_width, image_height);
                         gif.create_gif(filename, fps, quality, loop););
    }

private:
    int image_height;          // Rendered image height
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    Point3 center;             // Camera center
    Point3 pixel00_loc;        // Location of pixel 0, 0
    Vec3 pixel_delta_u;        // Offset to pixel to the right
    Vec3 pixel_delta_v;        // Offset to pixel below

    void initialize()
    {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focal_length;
        auto viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        Vec3 w = unit_vector(lookfrom - lookat);
        Vec3 u = unit_vector(cross(vup, w));
        Vec3 v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3 viewport_u = viewport_width * u;   // Vector across viewport horizontal edge
        Vec3 viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    std::vector<Color> render_pixels(const Scene &world)
    {
        std::vector<Color> buffer;
        if (antialiasing && samples_per_pixel > 0)
        {
            for (int j = 0; j < image_height; j++)
            {
                for (int i = 0; i < image_width; i++)
                {
                    Color pixel_color(0, 0, 0);
                    for (int sample = 0; sample < samples_per_pixel; sample++)
                    {
                        Ray r = get_ray(i, j);
                        pixel_color += ray_color(r, world);
                    }
                    buffer.push_back(pixel_samples_scale * pixel_color);
                }
            }
        }
        else
        {
            for (int j = 0; j < image_height; j++)
            {
                for (int i = 0; i < image_width; i++)
                {
                    auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
                    auto ray_direction = pixel_center - center;
                    Ray r(center, ray_direction);

                    buffer.push_back(ray_color(r, world));
                }
            }
        }
        return buffer;
    }

    Ray get_ray(int i, int j) const
    {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_square();
        auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    Vec3 sample_square() const
    {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return Vec3(random_float() - 0.5, random_float() - 0.5, 0);
    }

    Color ray_color(const Ray &r, const Scene &world) const
    {
        HitRecord rec;

        // If the ray hits an object in the scene
        if (world.hit(r, Interval(0, INFINITY), rec))
        {
            Color res_color(0, 0, 0);

            Color texture_color = rec.tex->value(rec.u, rec.v, rec.p);

            res_color += texture_color * world.ambient_color;

            for (const auto &light : world.lights)
            {
                // Calculate the direction to the light source
                Vec3 light_direction_vector = light->position - rec.p;
                Vec3 light_direction = unit_vector(light_direction_vector);

                Ray shadow_ray(rec.p, light_direction);

                // Distance to the light source
                float light_distance = (light_direction_vector).length();

                // Check if there is any object between the point and the light source
                if (!world.hit(shadow_ray, Interval(0.001, light_distance), HitRecord()))
                {
                    // Light contribution if not in shadow
                    const Vec3 L = light_direction;
                    const Vec3 V = unit_vector(-r.direction()); // Viewing direction is the opposite of ray direction
                    const Vec3 H = unit_vector(L + V);

                    float diffuse_factor = std::max(dot(rec.normal, L), 0.0f);
                    float specular_factor = std::pow(std::max(dot(rec.normal, H), 0.0f), rec.tex->specular_exponent);

                    Color diffuse_term = diffuse_factor * (texture_color * light->diffuse_color);
                    Color specular_term = specular_factor * light->specular_color;

                    res_color += diffuse_term + specular_term;
                }
            }

            return res_color;
        }

        // Background color
        Vec3 unit_direction = unit_vector(r.direction());
        auto t = 0.5 * (unit_direction.y() + 1.0);

        // return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.25, 0.5, 1.0);

        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
    }

    void rotate_around_look_at(float angle)
    {
        // Compute the radius of the circle
        Vec3 direction = lookfrom - lookat;
        float radius = direction.length();

        // Compute the new position using spherical coordinates
        float theta = std::atan2(direction.z(), direction.x()) + (angle * (M_PI / 180.0f));
        lookfrom[0] = lookat.x() + radius * std::cos(theta);
        lookfrom[2] = lookat.z() + radius * std::sin(theta);

        // Maintain the same y-coordinate for simplicity
        lookfrom[1] = lookat.y() + direction.y();
    }
};