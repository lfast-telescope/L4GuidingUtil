import astropy.units as u
import matplotlib.pyplot as plt

from L4_guiding_util.telescope_sim import (
    generate_star_gaussian,
    apply_telescope,
    apply_fiber,
    apply_relay,
    lfast1x_telescope_params,
    lfast1x_relay_params,
    lfast1x_fiber_params,
    default_sensor_params,
    capture_image,
)

if __name__ == "__main__":
    scale = 16
    region_len = 64 * scale
    pixel_scale = (
        default_sensor_params.pixel_pitch / scale * lfast1x_telescope_params.plate_scale
    )

    shape = (region_len, region_len)

    s = generate_star_gaussian(
        shape, fwhm=1.0 * u.arcsec, pixel_scale=pixel_scale, magnitude=5.0
    )

    apply_telescope(s, lfast1x_telescope_params)

    f = apply_fiber(
        s,
        (0, 0),
        lfast1x_fiber_params,
    )

    print(f"{f.fib_flux_ratio()=}")

    apply_relay(f, lfast1x_relay_params)

    c = capture_image(f, default_sensor_params, exposure=16 * u.ms)

    c.save_fits("generate_to_img.fits")

    c.show_img()
    plt.show()
