import numpy as np
from matplotlib import pyplot as plt
import astropy.modeling
import astropy.units as u
import astropy.io.fits as fits

from util import show_img

"""
TODO:
- figure out where input parameters should come from
- allow input parameters from different places
- Output to some kind of save file/directory
"""

# Other input params needed:
#   - star magnitude
#   - r_0
#   - subpixel angle
#   - region size

lfast1x_telescope_params = {
    'collecting_area':0.4368 * u.m**2,
    'plate_scale':77.34 * u.arcsec / u.mm,
}

def get_zero_point_flux(filter = "V"):
    return (3.630e-9*u.erg/u.s/u.cm**2/u.angstrom)\
        .to(u.ph/u.s/u.m**2/u.um, u.spectral_density(0.550*u.um))\
        * 0.086*u.um

def get_apparent_flux_over_area(magnitude=0.0, area=None, filter="V"):
    if area == None:
        area = lfast1x_telescope_params['collecting_area']
    psm = get_zero_point_flux(filter)
    return (psm * area * 10**(-magnitude/2.5)).to(u.ph/u.s)

def generate_star(magnitude, r_0, subpixel_angle, telescope_area, region_size):
    """Generates a star image as a Gaussian based on magnitude and seeing
    The image's units are """

    avg_wavelength = 560 * u.nm

    # Calculate fwhm limitation of seeing
    fwhm = (u.rad * 1.029 * avg_wavelength / r_0).to(u.arcsec)
    # Sigma of a Gaussian approximation of Airy disk given by fwhm
    sigma = (fwhm / 2.355)

    # Un-normalized Star model (subpixels)
    star_model = astropy.modeling.models.Gaussian2D(1.0,
        0.0,
        0.0,
        (sigma).value,
        (sigma).value
    )

    # Create an x/y range of um values over the image sensor
    x_min = -region_size[1] / 2
    x_max = x_min + region_size[1]
    x_range_arcsec = np.arange(x_min, x_max, 1) * subpixel_angle.value

    y_min = -region_size[0] / 2
    y_max = y_min + region_size[0]
    y_range_arcsec = np.arange(y_min, y_max, 1) * subpixel_angle.value

    # Create a mesh grid of um x um points
    x,y = np.meshgrid(x_range_arcsec, y_range_arcsec)

    # Create image from model
    star_img = star_model(x,y)

    # Get total incoming photons from magnitude and telescope area TODO: work out area into param
    photons_per_sec = get_apparent_flux_over_area(magnitude, telescope_area)

    # Normalize star img to photons per second
    star_img = star_img / star_img.sum() * photons_per_sec.value
    pixel_scale = u.pixel_scale(subpixel_angle / u.pixel)
    return star_img, pixel_scale

def load_star_img(fname, file_type):
    full_fname = f"{fname}.{file_type}"

    hdul = fits.open(full_fname)
    hdu = hdul[0]
    subpixel_angle = hdu.header['SUBPIXAN'] * u.arcsec
    data = hdu.data

    return data, subpixel_angle


def save_star_img(fname, file_type, star_img_data, subpixel_angle):
    full_fname = f"{fname}.{file_type}"

    hdu = fits.PrimaryHDU(data=star_img_data)

    hdu.header['SUBPIXAN'] = subpixel_angle.value
    hdul = fits.HDUList([hdu])

    # Throws OSError if file already exists, unless overwrite=True
    hdul.writeto(full_fname, overwrite=True)

    # fits.open(full_fname, mode)


if __name__ == "__main__":

    magnitude = 10
    r_0 = 10 * u.cm
    plate_scale = 77.34 * u.arcsec / u.mm
    telescope_area = 0.4368 * u.m**2
    pixel_pitch = 1.85 * u.um
    upscale = 16

    subpixel_angle = (pixel_pitch * plate_scale / upscale).to(u.arcsec)

    region_size = np.array([63,64])

    large_region_size = region_size * upscale
    
    # star_img, subpixel_angle = load_star_img("test", "fits")

    star_img = generate_star(
        magnitude,
        r_0, 
        subpixel_angle, 
        telescope_area, 
        large_region_size
    )

    # save_star_img("test", "fits", star_img, subpixel_angle)

    show_img(star_img, subpixel_angle, "photons/sec", "Star img")

