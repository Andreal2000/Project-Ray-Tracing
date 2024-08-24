#pragma once

#include "cuda_helper.h"
#include "managed.h"
#include "scene.h"
#include "output_image.h"
#include "gif.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <chrono>

#define MEASURE_DURATION(name, code_block)                                                   \
    {                                                                                        \
        std::clog << name;                                                                   \
        auto start = std::chrono::high_resolution_clock::now();                              \
        code_block auto stop = std::chrono::high_resolution_clock::now();                    \
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); \
        std::clog << " took " << duration.count() << "ms" << std::endl;                      \
    }

__global__ void render_curand_init(int max_x, int max_y, curandState *rand_state);

class Camera : public Managed
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

    void render(Scene **world, const OutputImage &output)
    {
        uint8_t *buffer;
        MEASURE_DURATION("Rendering",
                         initialize();

                         // Render our buffer
                         int num_pixels = image_height * image_width;
                         size_t buffer_size = num_pixels * 3 * sizeof(uint8_t);

                         // allocate buffer
                         checkCudaErrors(cudaMallocManaged((void **)&buffer, buffer_size));

                         render_pixels(world, buffer, 0););

        MEASURE_DURATION("Exporting", output.write_image(buffer, image_width, image_height););

        checkCudaErrors(cudaFree(buffer));
    }

    void render_gif(Scene **world, const std::string &filename, int duration, int fps, int quality, bool loop)
    {
        assert(quality >= 2 && quality <= 8);

        int num_frames = (duration * 0.001) * fps;

        uint8_t *buffer;
        MEASURE_DURATION("Rendering",

                         initialize();

                         Vec3 original_lookfrom = lookfrom;

                         // Render our buffer
                         int num_pixels = image_height * image_width;
                         size_t buffer_size = num_pixels * num_frames * 3 * sizeof(uint8_t);

                         // allocate buffer
                         checkCudaErrors(cudaMallocManaged((void **)&buffer, buffer_size));

                         for (int f = 0; f < num_frames; f++) {
                             lookfrom = original_lookfrom;

                             rotate_around_look_at((360.0 / num_frames) * f);

                             initialize();

                             render_pixels(world, buffer, f); }

        );

        MEASURE_DURATION("Exporting",
                         Gif gif(buffer, num_frames, image_width, image_height);
                         gif.create_gif(filename, fps, quality, loop););

        checkCudaErrors(cudaFree(buffer));
    }

private:
    int image_height;          // Rendered image height
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    Point3 center;             // Camera center
    Point3 pixel00_loc;        // Location of pixel 0, 0
    Vec3 pixel_delta_u;        // Offset to pixel to the right
    Vec3 pixel_delta_v;        // Offset to pixel below

    int tx = 8;
    int ty = 8;

    void initialize()
    {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto focal_length = (lookfrom - lookat).length();
        auto theta = vfov * M_PI / 180;
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

    inline void render_pixels(Scene **world, uint8_t *buffer, int frame_number)
    {
        dim3 blocks(image_width / tx + 1, image_height / ty + 1);
        dim3 threads(tx, ty);
        if (antialiasing && samples_per_pixel > 0)
        {
            // allocate random state
            curandState *rand_state;
            checkCudaErrors(cudaMalloc((void **)&rand_state, image_height * image_width * sizeof(curandState)));

            render_curand_init<<<blocks, threads>>>(image_width, image_height, rand_state);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            render_kernel_aa<<<blocks, threads>>>(buffer, this, world, frame_number, rand_state);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaFree(rand_state));
        }
        else
        {
            render_kernel<<<blocks, threads>>>(buffer, this, world, frame_number);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }

    __device__ Ray get_ray(int i, int j, curandState *rand_state) const
    {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_square(rand_state);
        auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }

    __device__ Vec3 sample_square(curandState *rand_state) const
    {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return Vec3(curand_uniform(rand_state) - 0.5, curand_uniform(rand_state) - 0.5, 0);
    }

    __device__ Color ray_color(const Ray &r, Scene *world)
    {
        HitRecord rec;

        // If the ray hits an object in the scene
        if (world->hit(r, Interval(0, INFINITY), rec))
        {
            Color res_color(0, 0, 0);

            Color texture_color = rec.tex->value(rec.u, rec.v, rec.p);

            res_color += texture_color * world->ambient_color;

            for (int i = 0; i < world->lights_size; i++)
            {
                Light *light = world->lights[i];

                // Calculate the direction to the light source
                Vec3 light_direction_vector = light->position - rec.p;
                Vec3 light_direction = unit_vector(light_direction_vector);

                Ray shadow_ray(rec.p, light_direction);

                // Distance to the light source
                float light_distance = (light_direction_vector).length();

                // Check if there is any object between the point and the light source
                if (!world->hit(shadow_ray, Interval(0.001, light_distance), HitRecord()))
                {
                    // Light contribution if not in shadow
                    const Vec3 L = light_direction;
                    const Vec3 V = unit_vector(-r.direction()); // Viewing direction is the opposite of ray direction
                    const Vec3 H = unit_vector(L + V);

                    float diffuse_factor = max(dot(rec.normal, L), 0.0f);
                    float specular_factor = std::pow(max(dot(rec.normal, H), 0.0f), rec.tex->specular_exponent);

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

    friend __global__ void render_kernel(uint8_t *buffer, Camera *cam, Scene **world, int frame_number);
    friend __global__ void render_kernel_aa(uint8_t *buffer, Camera *cam, Scene **world, int frame_number, curandState *rand_state);
};

__global__ void render_kernel(uint8_t *buffer, Camera *cam, Scene **world, int frame_number)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= cam->image_width) || (j >= cam->image_height))
        return;

    int pixel_index = (cam->image_width * cam->image_height) * frame_number + j * cam->image_width + i;
    Color col(0, 0, 0);
    auto pixel_center = cam->pixel00_loc + (i * cam->pixel_delta_u) + (j * cam->pixel_delta_v);
    auto ray_direction = pixel_center - cam->center;
    Ray r(cam->center, ray_direction);

    col = cam->ray_color(r, *world);

    uint8_t rgb[3];

    color_to_byte(col, rgb);

    buffer[pixel_index * 3 + 0] = rgb[0];
    buffer[pixel_index * 3 + 1] = rgb[1];
    buffer[pixel_index * 3 + 2] = rgb[2];
}

__global__ void render_curand_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curand_init(pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render_kernel_aa(uint8_t *buffer, Camera *cam, Scene **world, int frame_number, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= cam->image_width) || (j >= cam->image_height))
        return;

    int pixel_index = (cam->image_width * cam->image_height) * frame_number + j * cam->image_width + i;
    curandState local_rand_state = rand_state[pixel_index];
    Color col(0, 0, 0);
    for (int s = 0; s < cam->samples_per_pixel; s++)
    {
        Ray r = cam->get_ray(i, j, &local_rand_state);
        col += cam->ray_color(r, *world);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(cam->samples_per_pixel);

    uint8_t rgb[3];

    color_to_byte(col, rgb);

    buffer[pixel_index * 3 + 0] = rgb[0];
    buffer[pixel_index * 3 + 1] = rgb[1];
    buffer[pixel_index * 3 + 2] = rgb[2];
}
