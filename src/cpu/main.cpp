#include "utils.h"
#include "camera.h"
#include "scene.h"
#include "sphere.h"
#include "quad.h"
#include "model.h"

#define USE_BOUNDING_BOX true
// #define USE_BOUNDING_BOX false

void balls()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    auto ball = make_shared<ColorTexture>(Color(0.25, 0.25, 1));
    auto ground = make_shared<ColorTexture>(Color(0.25, 1, 0.25));
    // auto checker = make_shared<checker_texture>(0.32, Color(.2, .3, .1), Color(.9, .9, .9));

    // world.add_object(make_shared<Quad>(Point3(-50, -1, 50), Point3(0, 0, -100), Point3(100, 0, 0), ground));

    world.add_object(make_shared<Sphere>(Point3(0, -100, -10), 100, ground));
    world.add_object(make_shared<Sphere>(Point3(1, 0, -2), 1.5, ball));
    world.add_object(make_shared<Sphere>(Point3(-1, 1, -3), 2.5, ball));
    world.add_object(make_shared<Sphere>(Point3(3, 2, -2.5), 0.5, ball));

    world.add_light(make_shared<Light>(Point3(50, 50, 20), 0.75));
    world.add_light(make_shared<Light>(Point3(0, 34, -20), 0.45));
    world.add_light(make_shared<Light>(Point3(0, 0.25, 4.5), 0.10));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 1, 3);
    cam.lookat = Point3(0, 1, -3);
    cam.vup = Vec3(0, 1, 0);

    // cam.render(world, Netpbm("images/balls.ppm"));
    cam.render(world, Bitmap("images/balls.bmp"));
    // cam.render_gif(world, "balls.gif", 1 * 1000, 8, 8, true);

    // cam.render(world, Bitmap("output/cpu/ballsAABB.bmp"));
    // cam.render_gif(world, "output/cpu/balls.gif", 5 * 1000, 24, 8, true);
}

void earth()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    auto earth_texture = make_shared<ImageTexture>("assets/textures/earthmap.jpg");

    world.add_object(make_shared<Sphere>(Point3(0, 0, 0), 5.5, earth_texture));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 0, 10);
    cam.lookat = Point3(0, 0, 0);
    cam.vup = Vec3(0, 1, 0);

    // cam.render(world, Bitmap("images/earth.bmp"));
    cam.render_gif(world, "earth.gif", 5 * 1000, 24, 8, true);

    // cam.render(world, Bitmap("output/cpu/earthAABB.bmp"));
    // cam.render_gif(world, "output/cpu/earth.gif", 5 * 1000, 24, 8, true);
}

void eight_ball()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    auto eight_ball_texture = make_shared<ImageTexture>("assets/models/Eight ball/8ball2Txt.png");

    world.add_object(make_shared<Model>("assets/models/Eight ball/8ball.obj", eight_ball_texture));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 0, 3);
    cam.lookat = Point3(0, 0, 0);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/eight_ball.bmp"));
}

void obamium()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    world.add_object(make_shared<Model>("assets/models/obamium/obamium.obj", "assets/models/obamium/"));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 10;

    cam.vfov = 60;
    cam.lookfrom = Point3(0, 5, 10);
    cam.lookat = Point3(-2.5, 2.5, 2.5);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/obamium.bmp"));
    // cam.render_gif(world, "obamium.gif", 5 * 1000, 24, 8, true);
}

void crash_bandicoot()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    world.add_object(make_shared<Model>("assets/models/crashbandicoot/crashbandicoot.obj", "assets/models/crashbandicoot/"));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 10;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 100, 140);
    cam.lookat = Point3(0, 90, 0);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/crash_bandicoot.bmp"));
}

void king()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    world.add_object(make_shared<Model>("assets/models/King/King 2/king2u.obj", "assets/models/King/King 2/"));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 10;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 0, 1300);
    cam.lookat = Point3(0, -150, 0);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/king.bmp"));
}

void cornell_box()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    auto grey = make_shared<ColorTexture>(Color(0.73, 0.73, 0.73));
    auto red = make_shared<ColorTexture>(Color(0.65, 0.05, 0.05));
    auto green = make_shared<ColorTexture>(Color(0.12, 0.45, 0.15));
    auto white = make_shared<ColorTexture>(Color(0.73, 0.73, 0.73));

    world.add_object(make_shared<Quad>(Point3(555, 0, 0), Vec3(0, 0, 555), Vec3(0, 555, 0), red));
    world.add_object(make_shared<Quad>(Point3(0, 0, 555), Vec3(0, 0, -555), Vec3(0, 555, 0), green));
    world.add_object(make_shared<Quad>(Point3(0, 555, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), white));
    world.add_object(make_shared<Quad>(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 0, -555), white));
    world.add_object(make_shared<Quad>(Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 555, 0), white));

    world.add_object(make_shared<Sphere>(Point3(190, 90, 190), 90, grey));

    world.add_object(make_shared<Quad>(Point3(265, 0, 295), Vec3(165, 0, 0), Vec3(0, 330, 0), grey));
    world.add_object(make_shared<Quad>(Point3(265, 0, 295), Vec3(0, 0, 105), Vec3(0, 330, 0), grey));
    world.add_object(make_shared<Quad>(Point3(430, 0, 295), Vec3(0, 0, 105), Vec3(0, 330, 0), grey));
    world.add_object(make_shared<Quad>(Point3(265, 330, 295), Vec3(165, 0, 0), Vec3(0, 0, 105), grey));
    world.add_object(make_shared<Quad>(Point3(265, 0, 400), Vec3(165, 0, 0), Vec3(0, 330, 0), grey));

    world.add_light(make_shared<Light>(Point3(555 / 2, 550, 555 / 2), 1.5));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 38; // 40
    cam.lookfrom = Point3(278, 278, -800);
    cam.lookat = Point3(278, 278, 0);
    cam.vup = Vec3(0, 1, 0);

    // cam.render(world, Netpbm("images/cornell_box.ppm"));
    // cam.render(world, Bitmap("images/cornell_box.bmp"));

    // cam.render_gif(world, "cornell_box.gif", 1 * 1000, 8, 8, true);

    // cam.render(world, Bitmap("output/cpu/cornell_boxAABB.bmp"));
    // cam.render_gif(world, "output/cpu/cornell_box.gif", 5 * 1000, 24, 8, true);
}

void mario()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    world.add_object(make_shared<Model>("assets/models/Mario/mario.obj", "assets/models/Mario/"));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 10;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 100, 140);
    cam.lookat = Point3(0, 90, 0);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/mario.bmp"));
}

void spot()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    world.ambient_color = Color(1, 1, 1);

    auto spot_texture = make_shared<ImageTexture>("assets/models/spot/spot_texture.png");

    world.add_object(make_shared<Model>("assets/models/spot/spot_triangulated.obj", spot_texture));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 75;
    cam.lookfrom = Point3(0, 0.5, -1.5);
    cam.lookat = Point3(0, 0.25, 0.5);
    cam.vup = Vec3(0, 1, 0);

    // cam.render(world, Bitmap("images/spot.bmp"));
    // cam.render_gif(world, "images/spot.gif", 1 * 1000, 8, 8, true);

    cam.render(world, Bitmap("output/cpu/spotBB.bmp"));
    // cam.render_gif(world, "output/cpu/spot.gif", 5 * 1000, 24, 8, true);
}

void teapot()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    // world.ambient_color = Color(1, 1, 1);

    auto grey = make_shared<ColorTexture>(Color(0.5, 0.5, 0.5));
    auto checker = make_shared<CheckerTexture>(0.33f, Color(0, 0, 0), Color(0.8, 0.8, 0.8));

    world.add_object(make_shared<Model>("assets/models/teapot.obj", grey));
    world.add_object(make_shared<Sphere>(Point3(0, -100, 0), 100, checker));

    world.add_light(make_shared<Light>(Point3(0, 10, 10), 0.5));
    world.add_light(make_shared<Light>(Point3(0, 10, -10), 0.5));
    world.add_light(make_shared<Light>(Point3(10, 10, 0), 0.5));
    world.add_light(make_shared<Light>(Point3(-10, 10, 0), 0.5));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 90;
    cam.lookfrom = Point3(0, 2.5, 4);
    cam.lookat = Point3(0, 1.5, 0);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/teapot.bmp"));
}

void bunny()
{
    Scene world;

    world.use_bounding_box = USE_BOUNDING_BOX;

    // world.ambient_color = Color(1, 1, 1);

    auto grey = make_shared<ColorTexture>(Color(0.5, 0.5, 0.5));
    auto checker = make_shared<CheckerTexture>(0.33f, Color(0, 0, 0), Color(0.8, 0.8, 0.8));

    world.add_object(make_shared<Model>("assets/models/bunny.obj", grey));
    world.add_object(make_shared<Sphere>(Point3(0, -100, 0), 100.03, checker));

    world.add_light(make_shared<Light>(Point3(0, 1, 1), 0.5));
    world.add_light(make_shared<Light>(Point3(0, 1, -1), 0.5));
    world.add_light(make_shared<Light>(Point3(1, 1, 0), 0.5));
    world.add_light(make_shared<Light>(Point3(-1, 1, 0), 0.5));

    Camera cam;

    cam.aspect_ratio = 1;
    cam.image_width = 500;
    // cam.antialiasing = true;
    cam.samples_per_pixel = 64;

    cam.vfov = 70;
    cam.lookfrom = Point3(0, 0.15, 0.2);
    cam.lookat = Point3(0, 0.1, 0);
    cam.vup = Vec3(0, 1, 0);

    cam.render(world, Bitmap("images/bunny.bmp"));

    // cam.render(world, Bitmap("output/cpu/bunnyBB.bmp"));
    // cam.render_gif(world, "output/cpu/bunny.gif", 5 * 1000, 24, 8, true);
}

int main()
{
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

    // https://graphics.stanford.edu/data/3Dscanrep/
}
