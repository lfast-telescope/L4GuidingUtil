import numpy as np
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

MAGNITUDE = 10.0
NUM_FWHMS = 8
MIN_FWHM = 0.5
MAX_FWHM = 3.0
NUM_OFFSETS = 100
fname = f"sweep_offsets_{NUM_FWHMS}_{MIN_FWHM}-{MAX_FWHM}_{NUM_OFFSETS}.npy"


def get_fiber_data(fwhms, offsets, region_len):

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


def get_sensor_data(magnitude, fwhms, offsets, region_len):

    shape = (region_len, region_len)

    fiber_fluxes = np.zeros((len(fwhms), len(offsets)))
    fib_flux_unit = None
    fiber_ratios = np.zeros((len(fwhms), len(offsets)))

    sensor_snr = np.zeros((len(fwhms), len(offsets)))

    for i, fwhm in enumerate(fwhms):

        s = generate_star_gaussian(
            shape, fwhm=fwhm * u.arcsec, pixel_scale=pixel_scale, magnitude=magnitude
        )
        apply_telescope(s, lfast1x_telescope_params)

        for j, offset in enumerate(offsets):

            f = apply_fiber(s, (int(offset + 0.5), 0), lfast1x_fiber_params)

            fiber_fluxes[i, j] = f.fib_flux.value
            fiber_ratios[i, j] = f.fib_flux_ratio()

            if (i, j) == (0, 0):
                fib_flux_unit = f.fib_flux.unit

            apply_relay(f, lfast1x_relay_params)

            sen = capture_image(f, default_sensor_params, exposure=16 * u.ms)

            sensor_snr[i, j] = sen.snr

    return fiber_ratios, sensor_snr


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

    found_fiber_data = False
    found_snr_data = False
    try:
        with open("fiber_ratios_" + fname, "rb") as f:
            fiber_ratios = np.load(f)
            found_fiber_data = True
    except FileNotFoundError:
        pass
    try:
        with open("snrs_" + fname, "rb") as f:
            # fiber_ratios = np.load(f)
            snrs = np.load(f)
            found_snr_data = True
    except FileNotFoundError:
        pass

    if not found_snr_data:
        print("No stored data... generating...")
        # fiber_ratios = get_fiber_data(fwhms, offsets, region_len)

        fiber_ratios, snrs = get_sensor_data(10.0, fwhms, offsets, region_len)

        with open("fiber_ratios_" + fname, "wb") as f:
            np.save(f, fiber_ratios)
        with open("snrs_" + fname, "wb") as f:
            np.save(f, snrs)

        found_fiber_data = True
        found_snr_data = True

    if not found_fiber_data:
        print("No stored fiber data... generating...")
        fiber_ratios = get_fiber_data(fwhms, offsets, region_len)

        with open("fiber_ratios_" + fname, "wb") as f:
            np.save(f, fiber_ratios)
        found_fiber_data = True

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    offsets_arcsecs = offsets * pixel_scale.value
    for i, fwhm in enumerate(fwhms):
        ax1.plot(
            offsets_arcsecs,
            fiber_ratios[i, :],
            label=f"fwhm={str(round(fwhm.value, 1))}",
        )
    for i, fwhm in enumerate(fwhms):
        ax2.plot(
            offsets_arcsecs,
            snrs[i, :],
            label=f"fwhm={str(round(fwhm.value, 1))}",
        )
    ax1.grid()
    ax1.set_title("Ratio of flux down fiber with star offsets")
    ax1.set_xlabel("Star offset from fiber center, arcsec")
    ax1.set_ylabel("Ratio of flux down fiber")
    ax1.legend()
    ax2.grid()
    ax2.set_title(f"Sensor Image snr at from star offsets (Magnitude={MAGNITUDE})")
    ax2.set_xlabel("Star offset from fiber center, arcsec")
    ax2.set_ylabel("SNR of image signal to sensor read noise")
    ax2.legend()
    plt.show()
