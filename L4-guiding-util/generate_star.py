import astropy.units as u

from telescope_sim import generate_star_gaussian


if __name__ == "__main__":
    s = generate_star_gaussian(
        (64 * 16, 64 * 16),
        1.5 * u.arcsec,
        1 / 16 * 1.85 * 0.077 * u.arcsec / u.pix,
        10.0,
    )

    s.save_fits("test_star_img.fits")
