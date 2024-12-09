import numpy as np
import matplotlib.pyplot as plt
import astropy.nddata
import astropy.units as u
import sys

from L4_guiding_util.annotated_image import SensorImage, StarImage, SensorParams
from L4_guiding_util.telescope_sim import (
    inverse_capture,
    apply_telescope,
    lfast1x_telescope_params,
    load_png_img,
    apply_fiber,
    lfast1x_fiber_params,
    lfast1x_relay_params,
)


DEFAULT_FNAME = "test.png"

if __name__ == "__main__":

    fname = DEFAULT_FNAME

    if len(sys.argv) > 1:
        fname = sys.argv[1]

    img = load_png_img(fname)

    sensor_params = SensorParams(
        name="IMX183CLK-J",
        pixel_pitch=2.4 * u.um,
        read_noise=1.8 * u.electron,
        gain=1 / 0.4 * u.adu / u.electron,
        dark_current=0.062 * u.electron / u.s,
        bias=0.0 * u.adu,
        quantum_efficiency=0.5 * u.electron / u.ph,
    )

    sensor_img = SensorImage(img, 16 * u.ms, sensor_params)

    sensor_img.show_img()

    star_img = inverse_capture(sensor_img, 1.85 * u.um / 16, (80 * 16, 80 * 16), 10.0)
    apply_telescope(star_img, lfast1x_telescope_params, inverse=True)

    img = astropy.nddata.block_reduce(star_img.values, 6)
    star_img.values = img
    star_img.show_img()

    # plt.show()

    apply_telescope(star_img, lfast1x_telescope_params)
    fiber_img = apply_fiber(star_img, star_offsets=(0, 0), params=lfast1x_fiber_params)
    fiber_img.show_img(title=f"Fiber flux ratio: {fiber_img.fib_flux_ratio()}")
    plt.show()
