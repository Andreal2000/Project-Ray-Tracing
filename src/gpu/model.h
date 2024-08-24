#pragma once

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "hittable.h"
#include "vec3.h"

#include <iostream>
#include <memory>
#include <tuple>

using std::make_shared;
using std::shared_ptr;

struct Vec2
{
    float e[2];
    Vec2() : e{0, 0} {}
    Vec2(float e0, float e1) : e{e0, e1} {}

    __device__ float x() const { return e[0]; }
    __device__ float y() const { return e[1]; }
};

struct Triangle : public Managed
{
    Point3 v0, v1, v2;
    Vec3 normal;
    Vec2 uv0, uv1, uv2;
    Texture **tex;

    Triangle(Point3 v0, Point3 v1, Point3 v2, Vec3 normal, Vec2 uv0, Vec2 uv1, Vec2 uv2, Texture **tex)
        : v0(v0), v1(v1), v2(v2), normal(normal), uv0(uv0), uv1(uv1), uv2(uv2), tex(tex) {}

    __device__ bool hit(const Ray &r, const Interval &ray_t, HitRecord &rec) const
    {
        // Möller-Trumbore ray-triangle intersection algorithm
        const float EPSILON = 1e-8;
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = cross(r.direction(), edge2);
        float a = dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return false; // This ray is parallel to this triangle.

        float f = 1.0 / a;
        Vec3 s = r.origin() - v0;
        float u = f * dot(s, h);
        if (u < 0.0 || u > 1.0)
            return false;

        Vec3 q = cross(s, edge1);
        float v = f * dot(r.direction(), q);
        if (v < 0.0 || u + v > 1.0)
            return false;

        float t = f * dot(edge2, q);
        if (t > EPSILON && ray_t.surrounds(t))
        {
            rec.t = t;
            rec.p = r.at(rec.t);
            rec.normal = normal;
            rec.set_face_normal(r, normal);

            // Interpolate texture coordinates
            float w = 1.0 - u - v;
            rec.u = w * uv0.x() + u * uv1.x() + v * uv2.x();
            rec.v = w * uv0.y() + u * uv1.y() + v * uv2.y();

            rec.tex = *tex;

            return true;
        }
        return false;
    }
};

struct TriangleMesh
{
    void free()
    {
        for (int i = 0; i < triangles_size; i++)
        {
            cudaFree(triangles[i]);
        }
        cudaFree(triangles);

        for (int i = 0; i < textures.size(); i++)
        {
            Texture::free_texture(textures[i]);
            delete texture_files[i];
        }
    }

    Triangle **triangles;
    int triangles_size;

    std::vector<Texture **> textures;
    std::vector<TextureFile *> texture_files;
};

class Model : public Hittable
{
public:
    // Point3 center;

    __device__ Model(Triangle **triangles, int triangles_size) : triangles(triangles), triangles_size(triangles_size) { compute_bounding_box(); }

    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override
    {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;
        for (int i = 0; i < triangles_size; i++)
        {
            if (triangles[i]->hit(r, Interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    static TriangleMesh load_model(const std::string &obj_path, const std::string &mtl_search_path) { return load_model(obj_path, mtl_search_path, false, nullptr); }

    static TriangleMesh load_model(const std::string &obj_path, Texture **tex) { return load_model(obj_path, "", true, tex); }

    __device__ AABB bounding_box() const override { return bbox; }

private:
    static TriangleMesh load_model(const std::string &obj_path, const std::string &mtl_search_path, bool external_texture, Texture **tex)
    {
        tinyobj::ObjReader reader;
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = mtl_search_path;

        if (!reader.ParseFromFile(obj_path, reader_config))
        {
            if (!reader.Error().empty())
            {
                std::cerr << "[ERROR] TinyObjReader:\n"
                          << reader.Error();
            }
            exit(1);
        }

        if (!reader.Warning().empty())
        {
            std::cout << "[WARNING] TinyObjReader:\n"
                      << reader.Warning();
        }

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();
        auto &materials = reader.GetMaterials();

        // Calculate the total number of triangles
        int curr_triangle = 0;
        int num_triangles = 0;
        for (const auto &shape : shapes)
        {
            for (const auto &num_face_vertices : shape.mesh.num_face_vertices)
            {
                num_triangles += num_face_vertices - 2;
            }
        }

        std::vector<Texture **> textures;
        std::vector<TextureFile *> texture_files;

        std::map<int, Texture **> texture_map;
        for (const auto &mat : materials)
        {

            if (external_texture || mat.diffuse_texname.empty())
            {
                texture_map[&mat - &materials[0]] = tex;
            }
            else
            {
                Texture **image_texture;
                checkCudaErrors(cudaMalloc(&image_texture, sizeof(ImageTexture *)));

                TextureFile *texture_file = new TextureFile(mtl_search_path + mat.diffuse_texname);

                create_image_texture<<<1, 1>>>(image_texture, texture_file);
                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                textures.push_back(image_texture);
                texture_files.push_back(texture_file);

                texture_map[&mat - &materials[0]] = image_texture;
            }
        }

        Triangle **triangles;
        checkCudaErrors(cudaMallocManaged(&triangles, num_triangles * sizeof(Triangle)));

        for (const auto &shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                size_t fv = shape.mesh.num_face_vertices[f];
                int mat_id = shape.mesh.material_ids[f];

                std::vector<Point3> face_vertices;
                std::vector<Vec2> face_uvs;
                for (size_t v = 0; v < fv; v++)
                {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    Point3 vertex(
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]);
                    face_vertices.push_back(vertex);

                    if (!attrib.texcoords.empty())
                    {
                        Vec2 uv(
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]);
                        face_uvs.push_back(uv);
                    }
                    else
                    {
                        face_uvs.push_back(Vec2(0, 0));
                    }
                }

                if (face_vertices.size() >= 3)
                {
                    for (size_t v = 1; v < face_vertices.size() - 1; v++)
                    {
                        Point3 v0 = face_vertices[0];
                        Point3 v1 = face_vertices[v];
                        Point3 v2 = face_vertices[v + 1];

                        Vec3 normal = unit_vector(cross(v1 - v0, v2 - v0));

                        Vec2 uv0 = face_uvs[0];
                        Vec2 uv1 = face_uvs[v];
                        Vec2 uv2 = face_uvs[v + 1];

                        triangles[curr_triangle++] = new Triangle(v0, v1, v2, normal, uv0, uv1, uv2, mat_id == -1 ? tex : texture_map[mat_id]);
                    }
                }

                index_offset += fv;
            }
        }
        return TriangleMesh({triangles, num_triangles, textures, texture_files});
    }

    __device__ void compute_bounding_box()
    {
        Point3 min_point(INFINITY, INFINITY, INFINITY);
        Point3 max_point(-INFINITY, -INFINITY, -INFINITY);

        // for (const auto &vertex : vertices)
        for (int i = 0; i < triangles_size; i++)
        {
            Triangle *t = triangles[i];

            for (auto vertex : {t->v0, t->v1, t->v2})
            {
                max_point[0] = fmax(max_point[0], vertex[0]);
                min_point[0] = fmin(min_point[0], vertex[0]);

                max_point[1] = fmax(max_point[1], vertex[1]);
                min_point[1] = fmin(min_point[1], vertex[1]);

                max_point[2] = fmax(max_point[2], vertex[2]);
                min_point[2] = fmin(min_point[2], vertex[2]);
            }
        }

        bbox = AABB(min_point, max_point);
    }

    Triangle **triangles;
    int triangles_size;
    AABB bbox;
};