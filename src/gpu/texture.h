#pragma once

#include "color.h"
#include "texture_file.h"
#include "cuda_helper.h"

class Texture;

__global__ void delete_texture(Texture **texture);

class Texture
{
public:
    __device__ virtual ~Texture() {};

    __device__ virtual Color value(float u, float v, const Point3 &p) const = 0;

    static inline void free_texture(Texture **texture)
    {
        delete_texture<<<1, 1>>>(texture);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(texture));
    }

    float specular_exponent = 50;
};

__global__ void delete_texture(Texture **texture)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        delete *texture;
    }
}

// ColorTexture

__global__ void create_color_texture(Texture **texture, float red, float green, float blue);

class ColorTexture : public Texture
{
public:
    __device__ ColorTexture(const Color &albedo) : albedo(albedo) {}

    __device__ ColorTexture(float red, float green, float blue) : ColorTexture(Color(red, green, blue)) {}

    __device__ Color value(float u, float v, const Point3 &p) const override
    {
        return albedo;
    }

    static inline Texture **malloc_texture(const Color &albedo) { return malloc_texture(albedo.x(), albedo.y(), albedo.z()); }
    static inline Texture **malloc_texture(float red, float green, float blue)
    {
        Texture **color_texture;
        checkCudaErrors(cudaMalloc(&color_texture, sizeof(ColorTexture *)));

        create_color_texture<<<1, 1>>>(color_texture, red, green, blue);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        return color_texture;
    }

private:
    Color albedo;
};

__global__ void create_color_texture(Texture **texture, float red, float green, float blue)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *texture = new ColorTexture(red, green, blue);
    }
}

// CheckerTexture

__global__ void create_checker_texture(Texture **texture, float scale, Texture **even, Texture **odd);
__global__ void create_checker_texture(Texture **texture, float scale, float c1r, float c1g, float c1b, float c2r, float c2g, float c2b);

class CheckerTexture : public Texture
{
public:
    // Constructor with existing textures
    __device__ CheckerTexture(float scale, Texture &even, Texture &odd)
        : inv_scale(1.0 / scale), even(&even), odd(&odd) {}

    // Constructor with colors
    __device__ CheckerTexture(float scale, const Color &c1, const Color &c2)
        : inv_scale(1.0 / scale),
          even(new ColorTexture(c1)),
          odd(new ColorTexture(c2)) {}

    // Destructor
    __device__ ~CheckerTexture()
    {
        delete even;
        delete odd;
    }

    __device__ Color value(float u, float v, const Point3 &p) const override
    {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

    static inline Texture **malloc_texture(float scale, Texture **even, Texture **odd)
    {
        Texture **checker_texture;
        checkCudaErrors(cudaMalloc(&checker_texture, sizeof(CheckerTexture *)));

        create_checker_texture<<<1, 1>>>(checker_texture, scale, even, odd);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        return checker_texture;
    }

    static inline Texture **malloc_texture(float scale, const Color &c1, const Color &c2)
    {
        Texture **checker_texture;
        checkCudaErrors(cudaMalloc(&checker_texture, sizeof(CheckerTexture *)));

        create_checker_texture<<<1, 1>>>(checker_texture, scale, c1.x(), c1.y(), c1.z(), c2.x(), c2.y(), c2.z());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        return checker_texture;
    }

private:
    float inv_scale;
    Texture *even;
    Texture *odd;
};

__global__ void create_checker_texture(Texture **texture, float scale, Texture **even, Texture **odd)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *texture = new CheckerTexture(scale, **even, **odd);
    }
}
__global__ void create_checker_texture(Texture **texture, float scale, float c1r, float c1g, float c1b, float c2r, float c2g, float c2b)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *texture = new CheckerTexture(scale, Color(c1r, c1g, c1b), Color(c2r, c2g, c2b));
    }
}

// ImageTexture

__global__ void create_image_texture(Texture **texture, TextureFile *texture_file);

class ImageTexture : public Texture
{
public:
    __device__ ImageTexture(TextureFile *image) : image(image) {}

    __device__ Color value(float u, float v, const Point3 &p) const override
    {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image->height() <= 0)
            return Color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = Interval(0, 1).clamp(u);
        v = 1.0 - Interval(0, 1).clamp(v); // Flip V to image coordinates

        auto i = int(u * image->width());
        auto j = int(v * image->height());
        auto pixel = image->pixel_data(i, j);

        auto color_scale = 1.0 / 255.0;
        return Color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

    static inline Texture **malloc_texture(TextureFile *texture_file)
    {
        Texture **image_texture;
        checkCudaErrors(cudaMalloc(&image_texture, sizeof(ImageTexture *)));

        create_image_texture<<<1, 1>>>(image_texture, texture_file);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        return image_texture;
    }

private:
    TextureFile *image;
};

__global__ void create_image_texture(Texture **texture, TextureFile *texture_file)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *texture = new ImageTexture(texture_file);
    }
}