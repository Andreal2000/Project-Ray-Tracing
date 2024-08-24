#pragma once

#include "utils.h"
#include "color.h"

class OutputImage
{
public:
    virtual ~OutputImage() = default;

    OutputImage(std::string filename) : out(std::ofstream(filename)) {}

    virtual void write_image(std::vector<Color> &buffer, int image_width, int image_height) const = 0;

protected:
    mutable std::ofstream out;
};

class Netpbm : public OutputImage
{
public:
    Netpbm(const std::string &filename, bool binary = false) : OutputImage(filename), binary(binary)
    {
        if (binary)
        {
            out = std::ofstream(filename, std::ios::binary);
        }
    }

    void write_image(std::vector<Color> &buffer, int image_width, int image_height) const override
    {
        out << (binary ? "P6" : "P3") << "\n"
            << image_width << ' ' << image_height << "\n"
            << "255" << "\n";

        if (binary)
        {
            for (auto color : buffer)
            {
                uint8_t *rgb = color_to_byte(color);
                out << rgb[0] << rgb[1] << rgb[2];
            }
        }
        else
        {
            for (auto color : buffer)
            {
                uint8_t *rgb = color_to_byte(color);
                out << int(rgb[0]) << ' ' << int(rgb[1]) << ' ' << int(rgb[2]) << '\n';
            }
        }

        out.close();
    }

private:
    bool binary = false;
};

class Bitmap : public OutputImage
{
public:
    Bitmap(const std::string &filename) : OutputImage(filename) { out = std::ofstream(filename, std::ios::binary); }

    void write_image(std::vector<Color> &buffer, int image_width, int image_height) const override
    {
        BMPFileHeader fileHeader;
        BMPInfoHeader infoHeader;

        infoHeader.size = sizeof(BMPInfoHeader);
        infoHeader.width = image_width;
        infoHeader.height = image_height;
        fileHeader.offsetData = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
        fileHeader.fileSize = fileHeader.offsetData + image_width * image_height * 3;

        // Write headers
        out.write(reinterpret_cast<const char *>(&fileHeader), sizeof(fileHeader));
        out.write(reinterpret_cast<const char *>(&infoHeader), sizeof(infoHeader));

        for (int j = image_height - 1; j >= 0; j--)
        {
            for (int i = 0; i < image_width; i++)
            {
                uint8_t *rgb = color_to_byte(buffer[image_width * j + i]);

                out.put(rgb[2]); // blue
                out.put(rgb[1]); // green
                out.put(rgb[0]); // red
            }

            // BMP row padding to align to 4-byte boundary
            for (int p = 0; p < (4 - (image_width * 3) % 4) % 4; p++)
            {
                out.put(0);
            }
        }

        out.close();
    }

private:
// BMP file header structure
#pragma pack(push, 1)
    struct BMPFileHeader
    {
        unsigned short fileType{0x4D42}; // "BM"
        unsigned int fileSize{0};
        unsigned short reserved1{0};
        unsigned short reserved2{0};
        unsigned int offsetData{0};
    };

    // BMP info header structure
    struct BMPInfoHeader
    {
        unsigned int size{0};
        int width{0};
        int height{0};
        unsigned short planes{1};
        unsigned short bitCount{24}; // 24-bit color
        unsigned int compression{0};
        unsigned int sizeImage{0};
        int xPixelsPerMeter{0};
        int yPixelsPerMeter{0};
        unsigned int colorsUsed{0};
        unsigned int colorsImportant{0};
    };
#pragma pack(pop)
};
