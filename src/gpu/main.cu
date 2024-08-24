#include <iostream>
#include <curand_kernel.h>

#include "camera.h"
#include "sphere.h"
#include "quad.h"
#include "model.h"

#define ASSIGN_ELEMENTS(TYPE, array, ...)                       \
    {                                                           \
        TYPE *ptrs[] = {__VA_ARGS__};                           \
        for (int i = 0; i < sizeof(ptrs) / sizeof(TYPE *); i++) \
        {                                                       \
            array[i] = ptrs[i];                                 \
        }                                                       \
    }

#define USE_BOUNDING_BOX true
// #define USE_BOUNDING_BOX false

__global__ void free_scene(Hittable **d_objects, int objects_size, Texture **d_textures, int textures_size, Light **d_lights, int lights_size, Scene **d_world)
{
    for (int i = 0; i < objects_size; i++)
    {
        delete d_objects[i];
    }

    for (int i = 0; i < lights_size; i++)
    {
        delete d_lights[i];
    }

    for (int i = 0; i < textures_size; i++)
    {
        delete d_textures[i];
    }

    delete *d_world;
}

__global__ void create_balls(Hittable **d_objects, int objects_size, Texture **d_textures, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Texture, d_textures,
                        new ColorTexture(Color(0.25, 1, 0.25)),  // ground
                        new ColorTexture(Color(0.25, 0.25, 1))); // ball

        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Sphere(Point3(0, -100, -10), 100, d_textures[0]),
                        new Sphere(Point3(1, 0, -2), 1.5, d_textures[1]),
                        new Sphere(Point3(-1, 1, -3), 2.5, d_textures[1]),
                        new Sphere(Point3(3, 2, -2.5), 0.5, d_textures[1]));

        ASSIGN_ELEMENTS(Light, d_lights,
                        new Light(Point3(50, 50, 20), 0.75),
                        new Light(Point3(0, 34, -20), 0.45),
                        new Light(Point3(0, 0.25, 4.5), 0.10));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void balls()
{
    const int objects_size = 4;
    const int lights_size = 3;
    const int textures_size = 2;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    Texture **d_textures;
    checkCudaErrors(cudaMalloc(&d_textures, textures_size * sizeof(Texture *)));

    Light **d_lights;
    checkCudaErrors(cudaMalloc(&d_lights, lights_size * sizeof(Light *)));

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_balls<<<1, 1>>>(d_objects, objects_size, d_textures, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 1, 3);
    cam->lookat = Point3(0, 1, -3);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("balls.ppm"));
    cam->render(d_world, Bitmap("balls.bmp"));

    // cam->render_gif(d_world, "balls.gif", 1 * 1000, 8, 8, true);

    // cam->render(d_world, Bitmap("output/gpu/balls.bmp"));
    // cam->render_gif(d_world, "output/gpu/balls.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    checkCudaErrors(cudaFree(d_textures));
    checkCudaErrors(cudaFree(d_lights));
}

__global__ void create_earth(Hittable **d_objects, int objects_size, Texture **d_textures, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Sphere(Point3(0, 0, 0), 5.5, *d_textures));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void earth()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TextureFile *earthmap = new TextureFile("assets/textures/earthmap.jpg");
    Texture **d_textures = ImageTexture::malloc_texture(earthmap);

    Light **d_lights = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_earth<<<1, 1>>>(d_objects, objects_size, d_textures, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 0, 10);
    cam->lookat = Point3(0, 0, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("earth.ppm"));
    cam->render(d_world, Bitmap("earth.bmp"));

    // cam->render_gif(d_world, "earth.gif", 1 * 1000, 8, 8, true);

    // cam->render(d_world, Bitmap("output/gpu/earthAABB.bmp"));
    // cam->render_gif(d_world, "output/gpu/earth.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    Texture::free_texture(d_textures);
    delete earthmap;
}

__global__ void create_eight_ball(Hittable **d_objects, int objects_size, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void eight_ball()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TextureFile *texture_file = new TextureFile("assets/models/Eight ball/8ball2Txt.png");

    Texture **image_texture = ImageTexture::malloc_texture(texture_file);

    TriangleMesh eight_ball = Model::load_model("assets/models/Eight ball/8ball.obj", image_texture);

    Light **d_lights = nullptr;

    Texture **d_textures = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_eight_ball<<<1, 1>>>(d_objects, objects_size, eight_ball.triangles, eight_ball.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 0, 3);
    cam->lookat = Point3(0, 0, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("eight_ball.ppm"));
    cam->render(d_world, Bitmap("eight_ball.bmp"));

    // cam->render_gif(d_world, "eight_ball.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    eight_ball.free();

    Texture::free_texture(image_texture);
    delete texture_file;
}

__global__ void create_obamium(Hittable **d_objects, int objects_size, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void obamium()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TriangleMesh obamium = Model::load_model("assets/models/obamium/obamium.obj", "assets/models/obamium/");

    Light **d_lights = nullptr;

    Texture **d_textures = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_obamium<<<1, 1>>>(d_objects, objects_size, obamium.triangles, obamium.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 60;
    cam->lookfrom = Point3(0, 5, 10);
    cam->lookat = Point3(-2.5, 2.5, 2.5);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("obamium.ppm"));
    cam->render(d_world, Bitmap("obamium.bmp"));

    // cam->render_gif(d_world, "obamium.gif", 1 * 1000, 8, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    obamium.free();
}

__global__ void create_crash_bandicoot(Hittable **d_objects, int objects_size, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void crash_bandicoot()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TriangleMesh crash_bandicoot = Model::load_model("assets/models/crashbandicoot/crashbandicoot.obj", "assets/models/crashbandicoot/");

    Light **d_lights = nullptr;

    Texture **d_textures = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_crash_bandicoot<<<1, 1>>>(d_objects, objects_size, crash_bandicoot.triangles, crash_bandicoot.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 100, 140);
    cam->lookat = Point3(0, 90, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("crash_bandicoot.ppm"));
    cam->render(d_world, Bitmap("crash_bandicoot.bmp"));

    // cam->render_gif(d_world, "crash_bandicoot.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    crash_bandicoot.free();
}

__global__ void create_king(Hittable **d_objects, int objects_size, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void king()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TriangleMesh king = Model::load_model("assets/models/King/King 2/king2u.obj", "assets/models/King/King 2/");

    Light **d_lights = nullptr;

    Texture **d_textures = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_king<<<1, 1>>>(d_objects, objects_size, king.triangles, king.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 0, 1300);
    cam->lookat = Point3(0, -150, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("king.ppm"));
    cam->render(d_world, Bitmap("king.bmp"));

    // cam->render_gif(d_world, "king.gif", 1 * 1000, 8, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    king.free();
}

__global__ void create_cornell_box(Hittable **d_objects, int objects_size, Texture **d_textures, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Assign textures for the Cornell box
        ASSIGN_ELEMENTS(Texture, d_textures,
                        new ColorTexture(Color(0.73, 0.73, 0.73)),  // Grey for walls
                        new ColorTexture(Color(0.65, 0.05, 0.05)),  // Red for left wall
                        new ColorTexture(Color(0.12, 0.45, 0.15)),  // Green for right wall
                        new ColorTexture(Color(0.73, 0.73, 0.73))); // White for floor, ceiling, and back wall

        // Assign quads for the Cornell box
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Quad(Point3(555, 0, 0), Vec3(0, 0, 555), Vec3(0, 555, 0), d_textures[1]),  // Red left wall
                        new Quad(Point3(0, 0, 555), Vec3(0, 0, -555), Vec3(0, 555, 0), d_textures[2]), // Green right wall
                        new Quad(Point3(0, 555, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), d_textures[3]),  // White ceiling
                        new Quad(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 0, -555), d_textures[3]), // White floor
                        new Quad(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 555, 0), d_textures[3]),  // White back wall

                        new Sphere(Point3(190, 90, 190), 90, d_textures[0]), // Sphere

                        new Quad(Point3(265, 0, 295), Vec3(165, 0, 0), Vec3(0, 330, 0), d_textures[0]),   // Front face
                        new Quad(Point3(265, 0, 295), Vec3(0, 0, 105), Vec3(0, 330, 0), d_textures[0]),   // Left face
                        new Quad(Point3(430, 0, 295), Vec3(0, 0, 105), Vec3(0, 330, 0), d_textures[0]),   // Right face
                        new Quad(Point3(265, 330, 295), Vec3(165, 0, 0), Vec3(0, 0, 105), d_textures[0]), // Top face
                        new Quad(Point3(265, 0, 400), Vec3(165, 0, 0), Vec3(0, 330, 0), d_textures[0]));  // Back face

        // Assign lights
        ASSIGN_ELEMENTS(Light, d_lights,
                        new Light(Point3(555 / 2, 550, 555 / 2), 1.5)); // Ceiling light

        // Create the scene
        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void cornell_box()
{
    const int objects_size = 11;
    const int lights_size = 1;
    const int textures_size = 4;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    Texture **d_textures;
    checkCudaErrors(cudaMalloc(&d_textures, textures_size * sizeof(Texture *)));

    Light **d_lights;
    checkCudaErrors(cudaMalloc(&d_lights, lights_size * sizeof(Light *)));

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_cornell_box<<<1, 1>>>(d_objects, objects_size, d_textures, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 38; // 40
    cam->lookfrom = Point3(278, 278, -800);
    cam->lookat = Point3(278, 278, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("cornell_box.ppm"));
    cam->render(d_world, Bitmap("cornell_box.bmp"));

    // cam->render_gif(d_world, "cornell_box.gif", 1 * 1000, 8, 8, true);

    // cam->render(d_world, Bitmap("output/gpu/cornell_box.bmp"));
    // cam->render_gif(d_world, "output/gpu/cornell_box.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    checkCudaErrors(cudaFree(d_textures));
    checkCudaErrors(cudaFree(d_lights));
}

__global__ void create_mario(Hittable **d_objects, int objects_size, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void mario()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TriangleMesh mario = Model::load_model("assets/models/Mario/mario.obj", "assets/models/Mario/");

    Light **d_lights = nullptr;

    Texture **d_textures = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_mario<<<1, 1>>>(d_objects, objects_size, mario.triangles, mario.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 100, 140);
    cam->lookat = Point3(0, 90, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("mario.ppm"));
    cam->render(d_world, Bitmap("mario.bmp"));

    // cam->render_gif(d_world, "mario.gif", 1 * 1000, 8, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    mario.free();
}

__global__ void create_spot(Hittable **d_objects, int objects_size, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void spot()
{
    const int objects_size = 1;
    const int lights_size = 0;
    const int textures_size = 0;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    TextureFile *texture_file = new TextureFile("assets/models/spot/spot_texture.png");

    Texture **image_texture = ImageTexture::malloc_texture(texture_file);

    TriangleMesh spot = Model::load_model("assets/models/spot/spot_triangulated.obj", image_texture);

    Light **d_lights = nullptr;

    Texture **d_textures = nullptr;

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_spot<<<1, 1>>>(d_objects, objects_size, spot.triangles, spot.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 75;
    cam->lookfrom = Point3(0, 0.5, -1.5);
    cam->lookat = Point3(0, 0.25, 0.5);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("spot.ppm"));
    cam->render(d_world, Bitmap("spotR.bmp"));

    // cam->render_gif(d_world, "spot.gif", 1 * 1000, 8, 8, true);

    // cam->render(d_world, Bitmap("output/gpu/spotBB.bmp"));
    // cam->render_gif(d_world, "output/gpu/spot.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    spot.free();

    Texture::free_texture(image_texture);
    delete texture_file;
}

__global__ void create_teapot(Hittable **d_objects, int objects_size, Texture **d_textures, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        ASSIGN_ELEMENTS(Texture, d_textures,
                        // new ColorTexture(Color(0.73, 0.73, 0.73)));
                        new CheckerTexture(0.33, Color(0, 0, 0), Color(0.8, 0.8, 0.8)));

        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size),
                        new Sphere(Point3(0, -100, 0), 100, d_textures[0]));

        ASSIGN_ELEMENTS(Light, d_lights,
                        new Light(Point3(0, 10, 10), 0.5),
                        new Light(Point3(0, 10, -10), 0.5),
                        new Light(Point3(10, 10, 0), 0.5),
                        new Light(Point3(-10, 10, 0), 0.5));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        // (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void teapot()
{
    const int objects_size = 2;
    const int lights_size = 4;
    const int textures_size = 1;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    Texture **d_textures;
    checkCudaErrors(cudaMalloc(&d_textures, textures_size * sizeof(Texture *)));

    Texture **color_texture = ColorTexture::malloc_texture(0.5, 0.5, 0.5);

    TriangleMesh teapot = Model::load_model("assets/models/teapot.obj", color_texture);

    Light **d_lights;
    checkCudaErrors(cudaMalloc(&d_lights, lights_size * sizeof(Light *)));

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_teapot<<<1, 1>>>(d_objects, objects_size, d_textures, teapot.triangles, teapot.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 90;
    cam->lookfrom = Point3(0, 2.5, 4);
    cam->lookat = Point3(0, 1.5, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("teapot.ppm"));
    cam->render(d_world, Bitmap("teapot.bmp"));

    // cam->render_gif(d_world, "teapot.gif", 1 * 1000, 8, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    teapot.free();

    Texture::free_texture(color_texture);
}

__global__ void create_bunny(Hittable **d_objects, int objects_size, Texture **d_textures, Triangle **triangles, int triangles_size, Light **d_lights, int lights_size, Scene **d_world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        ASSIGN_ELEMENTS(Texture, d_textures,
                        // new ColorTexture(Color(0.73, 0.73, 0.73)));
                        new CheckerTexture(0.33, Color(0, 0, 0), Color(0.8, 0.8, 0.8)));

        ASSIGN_ELEMENTS(Hittable, d_objects,
                        new Model(triangles, triangles_size),
                        new Sphere(Point3(0, -100, 0), 100.03, d_textures[0]));

        ASSIGN_ELEMENTS(Light, d_lights,
                        new Light(Point3(0, 1, 1), 0.5),
                        new Light(Point3(0, 1, -1), 0.5),
                        new Light(Point3(1, 1, 0), 0.5),
                        new Light(Point3(-1, 1, 0), 0.5));
        // new Light(Point3(0, 1, 0), 1));

        *d_world = new Scene(d_objects, objects_size, d_lights, lights_size);

        // (*d_world)->ambient_color = Color(1, 1, 1);

        (*d_world)->use_bounding_box = USE_BOUNDING_BOX;
    }
}

void bunny()
{
    const int objects_size = 2;
    const int lights_size = 4;
    const int textures_size = 1;

    Hittable **d_objects;
    checkCudaErrors(cudaMalloc(&d_objects, objects_size * sizeof(Hittable *)));

    Texture **d_textures;
    checkCudaErrors(cudaMalloc(&d_textures, textures_size * sizeof(Texture *)));

    Texture **color_texture = ColorTexture::malloc_texture(0.5, 0.5, 0.5);

    TriangleMesh bunny = Model::load_model("assets/models/bunny.obj", color_texture);

    Light **d_lights;
    checkCudaErrors(cudaMalloc(&d_lights, lights_size * sizeof(Light *)));

    Scene **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable *)));

    create_bunny<<<1, 1>>>(d_objects, objects_size, d_textures, bunny.triangles, bunny.triangles_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera *cam = new Camera();

    cam->aspect_ratio = 1;
    cam->image_width = 500;
    // cam->antialiasing = true;
    cam->samples_per_pixel = 64;

    cam->vfov = 70;
    cam->lookfrom = Point3(0, 0.15, 0.2);
    cam->lookat = Point3(0, 0.1, 0);
    cam->vup = Vec3(0, 1, 0);

    // cam->render(d_world, Netpbm("bunny.ppm"));
    cam->render(d_world, Bitmap("bunny.bmp"));

    // cam->render_gif(d_world, "bunny.gif", 1 * 1000, 8, 8, true);

    // cam->render(d_world, Bitmap("output/gpu/bunnyBB.bmp"));
    // cam->render_gif(d_world, "output/gpu/bunny.gif", 5 * 1000, 24, 8, true);

    delete cam;
    free_scene<<<1, 1>>>(d_objects, objects_size, d_textures, textures_size, d_lights, lights_size, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    bunny.free();

    Texture::free_texture(color_texture);
}

int main()
{
    cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 1.5);

    switch (1)
    {
    case 1:
        balls();
        break;
    case 2:
        earth();
        break;
    case 3:
        eight_ball();
        break;
    case 4:
        obamium();
        break;
    case 5:
        crash_bandicoot();
        break;
    case 6:
        king();
        break;
    case 7:
        cornell_box();
        break;
    case 8:
        mario();
        break;
    case 9:
        spot();
        break;
    case 10:
        teapot();
        break;
    case 11:
        bunny();
        break;
    default:
        break;
    }

    cudaDeviceReset();
    return 0;
}
