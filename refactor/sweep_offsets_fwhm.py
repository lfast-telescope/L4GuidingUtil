import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from telescope_sim import (
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

NUM_FWHMS = 8
MIN_FWHM = 0.5
MAX_FWHM = 3.0
NUM_OFFSETS = 100
fname = f"sweep_offsets_{NUM_FWHMS}_{MIN_FWHM}-{MAX_FWHM}_{NUM_OFFSETS}.npy"


def get_data(fwhms, offsets, region_len):

    shape = (region_len, region_len)

    fiber_fluxes = np.zeros((len(fwhms), len(offsets)))
    fib_flux_unit = None
    fiber_ratios = np.zeros((len(fwhms), len(offsets)))

    for i, fwhm in enumerate(fwhms):

        s = generate_star_gaussian(
            shape, fwhm=fwhm * u.arcsec, pixel_scale=pixel_scale, magnitude=5.0
        )
        apply_telescope(s, lfast1x_telescope_params)

        for j, offset in enumerate(offsets):

            f = apply_fiber(s, (int(offset + 0.5), 0), lfast1x_fiber_params)

            fiber_fluxes[i, j] = f.fib_flux.value
            fiber_ratios[i, j] = f.fib_flux_ratio()

            if (i, j) == (0, 0):
                fib_flux_unit = f.fib_flux.unit

    return fiber_ratios


if __name__ == "__main__":

    fwhms = np.linspace(MIN_FWHM, MAX_FWHM, NUM_FWHMS) * u.arcsec

    scale = 16
    region_len = 64 * scale
    pixel_scale = (
        default_sensor_params.pixel_pitch / scale * lfast1x_telescope_params.plate_scale
    )

    # want ~4 arcsec offset lim
    offset_lim = (4 * u.arcsec / pixel_scale).to(u.pix).value
    offsets = np.linspace(-offset_lim, offset_lim, NUM_OFFSETS)

    found_data = True
    try:
        with open(fname, "rb") as f:
            fiber_ratios = np.load(f)
    except FileNotFoundError:
        found_data = False

    if not found_data:
        print("No stored data... generating...")
        fiber_ratios = get_data(fwhms, offsets, region_len)

        with open(fname, "wb") as f:
            np.save(fname, fiber_ratios)

    fig, ax1 = plt.subplots()

    offsets_arcsecs = offsets * pixel_scale.value

    for i, fwhm in enumerate(fwhms):
        ax1.plot(
            offsets_arcsecs,
            fiber_ratios[i, :],
            label=f"fwhm={str(round(fwhm.value, 1))}",
        )
    ax1.grid()
    ax1.set_title("Ratio of flux down fiber with star offsets")
    ax1.set_xlabel("Star offset from fiber center, arcsec")
    ax1.set_ylabel("Ratio of starlight down fiber")
    plt.legend()
    plt.show()
