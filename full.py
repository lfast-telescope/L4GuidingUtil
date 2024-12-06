import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from star_generation import generate_star, lfast1x_telescope_params
from fiber_mask_application import apply_fiber_mask, lfast1x_fiber_params, get_fiber_amount
from sensor_capture_sim import simulate_sensor_capture, test_sensor_params
from util import show_img


if __name__ == "__main__":

    upscale = 16
    magnitude = 10
    exposure = 16 * u.ms

    pixel_len = test_sensor_params['pixel_len']
    plate_scale = lfast1x_telescope_params['plate_scale']
    telescope_area = lfast1x_telescope_params['collecting_area']
    fiber_width = lfast1x_fiber_params['fiber_width']
    fiber_transmission = lfast1x_fiber_params['fiber_transmission']
    fiber_reflection = lfast1x_fiber_params['fiber_reflection']

    subpixel_angle = (pixel_len / upscale * plate_scale).to(u.arcsec)

    # x by y
    upscaled_region_size = np.array((64,64)) * upscale

    star_img, _ = generate_star(magnitude, 15*u.cm, subpixel_angle,
        telescope_area, upscaled_region_size)
    
    show_img(star_img, subpixel_angle)

    # TODO: Apply losses from primary, L1-4 lenses
    star_img *= (0.98**4 * 0.8)

    fiber_img, subpixel_len, fiber_transmit = apply_fiber_mask(star_img, subpixel_angle,
        plate_scale, 0.1*u.arcsec, 0, fiber_width, fiber_transmission, fiber_reflection)
    
    fiber_amt = get_fiber_amount(star_img, fiber_transmit)
    print(f"{fiber_amt=}")

    show_img(fiber_img, subpixel_len)

    # TODO: Apply losses from relay lenses
    fiber_img *= (0.98**4) * 0.98

    quantized_img = simulate_sensor_capture(fiber_img, subpixel_len, exposure,
        test_sensor_params)

    show_img(quantized_img, origin = "corner")
    plt.show()