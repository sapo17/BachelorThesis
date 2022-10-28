
import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


def plot_list(images, title=None):
    fig, axs = plt.subplots(1, len(images), figsize=(18, 3))
    for i in range(len(images)):
        axs[i].imshow(mi.util.convert_to_bitmap(images[i]))
        axs[i].axis('off')
    if title is not None:
        plt.suptitle(title)


def create_sensors(sensor_count, translate, rotate, rotationAngle, target, origin, up, fov, render_res):
    sensors = []
    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.translate(translate)
                                            .rotate([0, 1, 0], angle)
                                            .rotate(rotate, rotationAngle)
                                            .look_at(target=target, origin=origin, up=up),
            'film': {
                'type': 'hdrfilm',
                'width': render_res,
                'height': render_res,
                'filter': {'type': 'mitchell'},
                'pixel_format': 'rgba'
            }
        }))
    return sensors


# Rendering resolution
render_res = 1024

# Number of sensors
sensor_count = 15

translate = [0, 0.5, 0]
target = [0, 0, 0]
origin = [0, 0, -3]
up = [0, 1, 0]
fov = 60
rotate = [1, 0, 0]
angle = 0
sensors = create_sensors(sensor_count, translate, rotate,
                         angle, target, origin, up, fov, render_res)

rotate = [1, 0, 0]
angle = 20
sensors += create_sensors(sensor_count, translate, rotate,
                          angle, target, origin, up, fov, render_res)

rotate = [1, 0, 0]
angle = 40
sensors += create_sensors(sensor_count, translate, rotate,
                          angle, target, origin, up, fov, render_res)

rotate = [1, 0, 0]
angle = -20
sensors += create_sensors(sensor_count, translate, rotate,
                          angle, target, origin, up, fov, render_res)

rotate = [1, 0, 0]
angle = -30
sensors += create_sensors(sensor_count, translate, rotate,
                          angle, target, origin, up, fov, render_res)


scene_ref = mi.load_file(
    'python/mitsuba3-tutorial/scenes/cbox-sch-modified/simple-monkey.xml')
ref_images = [mi.render(scene_ref, sensor=sensor, spp=64)
              for sensor in sensors]


# save images
for idx, ref in enumerate(ref_images):
    output_path = 'python/reference-image-creator/output/'
    image_name = output_path + str(idx) + '.png'
    mi.util.write_bitmap(image_name, ref)
