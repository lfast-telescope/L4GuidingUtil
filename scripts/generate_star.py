import astropy.units as u

from L4_guiding_util.telescope_sim import generate_star_gaussian


if __name__ == "__main__":
    s = generate_star_gaussian(
        (64 * 16, 64 * 16),
        1.0 * u.arcsec,
        1 / 16 * 1.85 * 0.077 * u.arcsec / u.pix,
        10.0,
    )

    s.save_fits("test_star_img.fits")
