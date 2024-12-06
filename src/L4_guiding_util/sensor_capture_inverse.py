import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
import scipy.interpolate
import astropy.units as u

from util import show_img, translate_img
from fiber_mask_application import apply_fiber_mask, lfast1x_fiber_params, get_fiber_amount
from star_generation import lfast1x_telescope_params, get_apparent_flux_over_area
from sensor_capture_sim import simulate_sensor_capture, test_sensor_params

def invert_sensor_capture(img_data, subpixel_len, exposure, im_sensor_param):
    """Take a captured image and create an upsampled image from it"""

    # Ideas: 0 out bias? Use a gaussian based on noise, etc?
    # For now, just upsample


def load_png(png_fname):
    img = Image.open(png_fname)

    img_data = np.array(img.getdata()).reshape((img.size[1], img.size[0]))

    # TODO: get txt annotations wiht exposure time, pixel size, etc

    return img_data

def get_image_center(img_data):
    # Threshold data to isolate centroid calculation to star
    min_pix = np.min(img_data)
    max_pix = np.max(img_data)

    d_pix = max_pix - min_pix
    thresh = min_pix + int(d_pix * .1)

    img_data = np.where(img_data < thresh, 0, img_data)

    # Iterative, take centroid, then make the window smaller, etc etc?
    # Or just clip...
    # Maybe have some standard ways to index images?

    x,y = np.meshgrid(
        np.arange(0, img_data.shape[1]),
        np.arange(0, img_data.shape[0])
    )

    x_center = np.sum(img_data * x) / img_data.sum()
    y_center = np.sum(img_data * y) / img_data.sum()

    return x_center, y_center

def upscale_image(img_data, pixel_len, subpixel_len):
    """Perform interpolation to scale image up to desired size"""

    # TODO: upscale specifically based on desired pixel len?

    x_min = 0
    x_max = img_data.shape[1]
    y_min = 0
    y_max = img_data.shape[0]

    subpixel_x_max = int(x_max * pixel_len / subpixel_len)
    subpixel_y_max = int(y_max * pixel_len / subpixel_len)

    fit_points = [np.linspace(0,y_max, y_max), np.linspace(0, x_max, x_max)]
    ut, vt = np.meshgrid(np.linspace(0,y_max, subpixel_y_max), np.linspace(0, x_max, subpixel_x_max))
    test_points = np.array([ut.ravel(), vt.ravel()]).T

    interp = scipy.interpolate.RegularGridInterpolator(fit_points, img_data)
    upscaled_img = interp(test_points, method="linear").reshape(subpixel_x_max, subpixel_y_max).T

    return upscaled_img, subpixel_len

def crop_image(img_data, centroid, region_shape):
    x_min = int(centroid[0] - region_shape[1] // 2 + 0.5)
    x_max = x_min + region_shape[1]
    y_min = int(centroid[1] - region_shape[0] // 2 + 0.5)
    y_max = y_min + region_shape[0]

    return img_data[y_min:y_max, x_min:x_max]

if __name__ == "__main__":

    fiber_amts = {}

    for exp in ['16ms']:

        if exp == '16ms':
            exposure = 16 * u.ms
        if exp == '100ms':
            exposure = 100 * u.ms

        img = load_png(f"test_img_exp_{exp}.png") * u.ph
        img_data = img.value

        show_img(img_data, title=f"Original Image: {exp}", origin="corner")

        im_pixel_size = 2.4 * u.um
        
        final_pixel_len = 1.85 * u.um
        desired_subpixel_size = final_pixel_len / 16
        upscale = int(im_pixel_size / desired_subpixel_size)

        upscaled_img, subpixel_len = upscale_image(img_data, im_pixel_size, desired_subpixel_size)
        
        xc, yc = get_image_center(upscaled_img)

        centered_img = crop_image(upscaled_img, (xc, yc), (64*upscale,64*upscale))

        plate_scale = lfast1x_telescope_params['plate_scale']

        subpixel_angle = (subpixel_len * plate_scale).to(u.arcsec)

        # normalize image.
        magnitude = 10.0
        expected_incidence = get_apparent_flux_over_area(magnitude)

        star_img = centered_img / centered_img.sum() * expected_incidence


        fiber_width = lfast1x_fiber_params['fiber_width']
        fiber_transmission = lfast1x_fiber_params['fiber_transmission']
        fiber_reflection = lfast1x_fiber_params['fiber_reflection']

        fiber_img, subpixel_len, fiber_tx = apply_fiber_mask(star_img, subpixel_angle, plate_scale,
            0, 0, fiber_width, fiber_transmission, fiber_reflection)
        
        fiber_amts[exp] = get_fiber_amount(star_img, fiber_tx)

        show_img(fiber_img, subpixel_len, data_unit="flux photons/sec", title=f"Fiber applied: {exp}")

        quantized_img = simulate_sensor_capture(fiber_img, subpixel_len, exposure,
            im_sensor_param=test_sensor_params)


        show_img(quantized_img, final_pixel_len, data_unit="ADU counts/2^16", title=f"Sensor Image: {exp}", origin = "center")
    
    print(f"{fiber_amts=}")

    plt.show()

# TODO: Normalize upscaled image to expected magnitude