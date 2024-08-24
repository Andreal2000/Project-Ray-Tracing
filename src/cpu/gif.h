#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class Gif
{
public:
    Gif(const std::vector<std::vector<Color>> &frames, int image_width, int image_height)
        : image_width(image_width), image_height(image_height)
    {
        for (const auto &frame : frames)
        {
            std::vector<RGB> frame_bytes;
            for (const Color &color : frame)
            {
                uint8_t *color_byte = color_to_byte(color);
                frame_bytes.push_back(RGB(color_byte[0], color_byte[1], color_byte[2]));
            }
            this->frames.push_back(frame_bytes);
        }
    }

    // Function to create an example animated GIF
    void create_gif(const std::string &filename, int fps, int quality, bool loop)
    {
        const int delay = (1.0 / fps) * 100;

        // Collect all unique colors from all frames
        std::vector<RGB> colors;
        for (const auto &frame : frames)
        {
            colors.insert(colors.end(), frame.begin(), frame.end());
        }

        // Quantize the colors to a palette of 256 colors
        std::vector<RGB> palette = median_cut_quantize(colors, std::pow(2, quality));

        std::ofstream out(filename, std::ios::binary);

        // Write GIF Header
        out.write("GIF89a", 6);

        // Write Logical Screen Descriptor
        write_word(out, image_width);
        write_word(out, image_height);
        write_byte(out, 0xF0 | ((quality - 1) & 0x07)); // 11110111: Global Color Table Flag set, bit color resolution
        write_byte(out, 0);                             // Background Color Index
        write_byte(out, 0);                             // Pixel Aspect Ratio

        // Write Global Color Table
        for (const auto &color : palette)
        {
            write_byte(out, color.r);
            write_byte(out, color.g);
            write_byte(out, color.b);
        }

        if (loop)
        {
            write_byte(out, 0x21);        // Extension Introducer (introduced by an ASCII exclaimation point '!')
            write_byte(out, 0xFF);        // Application Extension
            write_byte(out, 0x0B);        // Size of block including application name and verification bytes (always 11)
            out.write("NETSCAPE2.0", 11); // 8-byte application name plus 3 verification bytes
            write_byte(out, 3);           // Number of bytes in the following sub-block
            write_byte(out, 1);           // Index of the current data sub-block (always 1 for the NETSCAPE block)
            write_word(out, 0);           // Loop count (0 for infinite)
            write_byte(out, 0);           // End of the sub-block chain for the Application Extension block
        }

        for (const auto &frame : frames)
        {
            // Write Graphics Control Extension
            write_byte(out, 0x21);  // Extension Introducer
            write_byte(out, 0xF9);  // Graphics Control Label
            write_byte(out, 0x04);  // Block Size
            write_byte(out, 0x04);  // 00001000: Disposal Method (none), User Input Flag (not set), Transparent Color Flag (not set)
            write_word(out, delay); // Delay Time in hundredths of a second
            write_byte(out, 0);     // Transparent Color Index
            write_byte(out, 0);     // Block Terminator

            // Write Image Descriptor
            write_byte(out, 0x2C); // Image Separator
            write_word(out, 0);    // Image Left Position
            write_word(out, 0);    // Image Top Position
            write_word(out, image_width);
            write_word(out, image_height);
            write_byte(out, 0x00); // No Local Color Table, no interlace

            // Map frame colors to palette indices
            std::vector<uint8_t> indices;
            indices.reserve(image_width * image_height); // Reserve space for efficiency
            for (RGB color : frame)
            {
                indices.push_back(find_closest_palette_color(color, palette));
            }

            // Compress the image data using LZW
            std::vector<uint8_t> compressed_data = lzw_encode(indices, quality);

            // Write LZW Minimum Code Size
            write_byte(out, quality);

            // Write Image Data in sub-blocks
            for (size_t i = 0; i < compressed_data.size(); i += 255)
            {
                size_t block_size = std::min(compressed_data.size() - i, static_cast<size_t>(255));
                write_byte(out, static_cast<uint8_t>(block_size));
                out.write(reinterpret_cast<const char *>(&compressed_data[i]), block_size);
            }

            // End of image data
            write_byte(out, 0);
        }

        // Write GIF Trailer
        write_byte(out, 0x3B); // Trailer

        out.close();
    }

private:
    // Function to write a single byte
    void write_byte(std::ofstream &out, uint8_t byte)
    {
        out.put(static_cast<char>(byte));
    }

    // Function to write a 16-bit word in little-endian format
    void write_word(std::ofstream &out, uint16_t word)
    {
        write_byte(out, word & 0xFF);
        write_byte(out, (word >> 8) & 0xFF);
    }

    struct RGB
    {
        uint8_t r, g, b;
        RGB(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}

        uint8_t operator[](int i) const
        {
            assert(i >= 0 && i < 3);
            if (i == 0)
                return r;
            if (i == 1)
                return g;
            return b;
        }
    };

    struct Box
    {
        Box(const std::vector<RGB> &colors) : colors(colors)
        {
            rmin = gmin = bmin = 255;
            rmax = gmax = bmax = 0;

            for (const auto &color : colors)
            {
                if (color.r < rmin)
                    rmin = color.r;
                if (color.r > rmax)
                    rmax = color.r;
                if (color.g < gmin)
                    gmin = color.g;
                if (color.g > gmax)
                    gmax = color.g;
                if (color.b < bmin)
                    bmin = color.b;
                if (color.b > bmax)
                    bmax = color.b;
            }
        }

        std::vector<RGB> colors;
        int rmin, rmax, gmin, gmax, bmin, bmax;
    };

    void median_cut(Box &box, int depth, std::vector<RGB> &palette)
    {
        if (depth == 0 || box.colors.size() <= 1)
        {
            // Calculate the average color of the box
            uint64_t rsum = 0, gsum = 0, bsum = 0;
            for (const auto &color : box.colors)
            {
                rsum += color.r;
                gsum += color.g;
                bsum += color.b;
            }

            RGB avg_color = {
                static_cast<uint8_t>(rsum / box.colors.size()),
                static_cast<uint8_t>(gsum / box.colors.size()),
                static_cast<uint8_t>(bsum / box.colors.size())};

            palette.push_back(avg_color);
            return;
        }

        int r_range = box.rmax - box.rmin;
        int g_range = box.gmax - box.gmin;
        int b_range = box.bmax - box.bmin;

        int split_component = 0;
        if (g_range > r_range && g_range >= b_range)
            split_component = 1;
        else if (b_range > r_range && b_range >= g_range)
            split_component = 2;

        // Sort colors by the chosen component
        std::sort(box.colors.begin(), box.colors.end(), [&](const RGB &a, const RGB &b)
                  { return a[split_component] < b[split_component]; });

        // Split the box at the median
        size_t median_index = box.colors.size() / 2;
        std::vector<RGB> left_colors(box.colors.begin(), box.colors.begin() + median_index);
        std::vector<RGB> right_colors(box.colors.begin() + median_index, box.colors.end());

        Box left_box(left_colors);
        Box right_box(right_colors);

        median_cut(left_box, depth - 1, palette);
        median_cut(right_box, depth - 1, palette);
    }

    std::vector<RGB> median_cut_quantize(const std::vector<RGB> &colors, int num_colors)
    {
        std::vector<RGB> palette;
        Box initial_box(colors);
        median_cut(initial_box, std::log2(num_colors), palette);
        return palette;
    }

    // Function to map a color to the nearest color in the palette
    uint8_t find_closest_palette_color(const RGB &color, const std::vector<RGB> &palette)
    {
        uint8_t closest_index = 0;
        int closest_distance = std::numeric_limits<int>::max();
        for (size_t i = 0; i < palette.size(); i++)
        {
            int dr = color[0] - palette[i][0];
            int dg = color[1] - palette[i][1];
            int db = color[2] - palette[i][2];
            int distance = dr * dr + dg * dg + db * db;
            if (distance < closest_distance)
            {
                closest_distance = distance;
                closest_index = i;
            }
        }
        return closest_index;
    }

    // LZW encoding function
    std::vector<uint8_t> lzw_encode(const std::vector<uint8_t> &indices, int code_size)
    {
        int clear_code = 1 << code_size;
        int end_code = clear_code + 1;
        int next_code = end_code + 1;
        int code_bits = code_size + 1;

        std::map<std::vector<uint8_t>, int> dictionary;
        for (int i = 0; i < clear_code; i++)
        {
            dictionary[{static_cast<uint8_t>(i)}] = i;
        }

        std::vector<uint8_t> result;
        int bit_buffer = 0;
        int bit_count = 0;

        auto output_code = [&](int code)
        {
            bit_buffer |= (code << bit_count);
            bit_count += code_bits;
            while (bit_count >= 8)
            {
                result.push_back(bit_buffer & 0xFF);
                bit_buffer >>= 8;
                bit_count -= 8;
            }
        };

        output_code(clear_code);

        std::vector<uint8_t> w;
        for (uint8_t k : indices)
        {
            std::vector<uint8_t> wk = w;
            wk.push_back(k);
            if (dictionary.find(wk) != dictionary.end())
            {
                w = wk;
            }
            else
            {
                output_code(dictionary[w]);
                if (next_code < 4096)
                {
                    dictionary[wk] = next_code++;
                    if (next_code > (1 << code_bits) && code_bits < 12)
                    {
                        ++code_bits;
                    }
                }
                else
                {
                    output_code(clear_code);

                    dictionary.clear();
                    for (int i = 0; i < clear_code; i++)
                    {
                        dictionary[{static_cast<uint8_t>(i)}] = i;
                    }

                    next_code = end_code + 1;
                    code_bits = code_size + 1;
                }
                w = {k};
            }
        }

        if (!w.empty())
        {
            output_code(dictionary[w]);
        }

        output_code(end_code);

        if (bit_count > 0)
        {
            result.push_back(bit_buffer & 0xFF);
        }

        return result;
    }

    std::vector<std::vector<RGB>> frames;
    int image_width;
    int image_height;
};