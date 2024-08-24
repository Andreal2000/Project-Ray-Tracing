#pragma once

#include "utils.h"
#include "texture_file.h"

class Texture
{
public:
    virtual ~Texture() = default;

    virtual Color value(float u, float v, const Point3 &p) const = 0;

    float specular_exponent = 50;
};

class ColorTexture : public Texture
{
public:
    ColorTexture(const Color &albedo) : albedo(albedo) {}

    ColorTexture(float red, float green, float blue) : ColorTexture(Color(red, green, blue)) {}

    Color value(float u, float v, const Point3 &p) const override
    {
        return albedo;
    }

private:
    Color albedo;
};

class CheckerTexture : public Texture
{
public:
    CheckerTexture(float scale, shared_ptr<Texture> even, shared_ptr<Texture> odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

    CheckerTexture(float scale, const Color &c1, const Color &c2)
        : inv_scale(1.0 / scale),
          even(make_shared<ColorTexture>(c1)),
          odd(make_shared<ColorTexture>(c2)) {}

    Color value(float u, float v, const Point3 &p) const override
    {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    float inv_scale;
    shared_ptr<Texture> even;
    shared_ptr<Texture> odd;
};

class ImageTexture : public Texture
{
public:
    ImageTexture(const std::string filename) : image(filename) {}

    Color value(float u, float v, const Point3 &p) const override
    {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0)
            return Color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = Interval(0, 1).clamp(u);
        v = 1.0 - Interval(0, 1).clamp(v); // Flip V to image coordinates

        auto i = int(u * image.width());
        auto j = int(v * image.height());
        auto pixel = image.pixel_data(i, j);

        auto color_scale = 1.0 / 255.0;
        return Color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

private:
    TextureFile image;
};