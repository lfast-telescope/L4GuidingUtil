import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
import astropy.io.fits as fits
from scipy.ndimage import convolve

from star_generation import load_star_img
from util import show_img, translate_img

lfast1x_fiber_params = {
    'fiber_width': 18*u.um,
    'fiber_transmission': 0.98,
    'fiber_reflection': 0.02,
}

def create_fiber_mask(region_shape, subpixel_len,
    fiber_width, fiber_transmission, fiber_reflection):

    x_min = -region_shape[1] / 2
    x_max = x_min + region_shape[1]
    x_range_um = np.arange(x_min, x_max, 1) * subpixel_len.value

    y_min = -region_shape[0] / 2
    y_max = y_min + region_shape[0]
    y_range_um = np.arange(y_min, y_max, 1) * subpixel_len.value

    fiber_radius = (fiber_width / 2).to(u.um)

    x,y = np.meshgrid(x_range_um, y_range_um)

    fiber = x*x + y*y <= fiber_radius.value**2

    fiber_mask = np.ones(region_shape)
    fiber_mask[fiber] = fiber_reflection

    fiber_allow = np.zeros(region_shape)
    fiber_allow[fiber] = fiber_transmission

    return fiber_mask, fiber_allow


def apply_fiber_mask(star_img: np.ndarray, subpixel_angle, plate_scale,
    star_offset_x, star_offset_y, 
    fiber_width, fiber_transmission, fiber_reflection):
    
    subpixel_len = (subpixel_angle / plate_scale).to(u.um)

    arcsec_length_equiv = (u.arcsec, u.mm, lambda x: x / 77.34, lambda x: x * 77.34)

    # if star_offset_x is float/int, use pixels? else, convert to subpix
    if type(star_offset_x) in [int, float]:
        offset_x = int(star_offset_x + 0.5)
    elif type(star_offset_x) == u.Quantity:
        offset_x = star_offset_x.to(u.um, [arcsec_length_equiv])/subpixel_len
        offset_x = int(offset_x.value + 0.5)
    if type(star_offset_y) in [int, float]:
        offset_y = int(star_offset_y + 0.5)
    elif type(star_offset_y) == u.Quantity:
        offset_y = star_offset_y.to(u.um, [arcsec_length_equiv])/subpixel_len
        offset_y = int(offset_y.value + 0.5)


    star_img = translate_img(star_img, offset_x, offset_y)

    fiber_mask, fiber_allow = create_fiber_mask(star_img.shape, subpixel_len, 
        fiber_width, fiber_transmission, fiber_reflection)

    masked_star_img = star_img * fiber_mask
    fiber_transmit = float((star_img * fiber_allow).sum()) * u.ph / u.s

    return masked_star_img, subpixel_len, fiber_transmit

# TODO: Finish this
def analyze_fiber_position(star_img, subpixel_angle, plate_scale):
    # fx = fiber_allow.shape[1]//2
    # fy = fiber_allow.shape[0]//2
    # fiber_allow_small = fiber_allow[(fy-80):(fx+80),(fx-80):(fx+80)]

    # show_img(fiber_allow_small)
    
    # # Try convolving fiber mask with image to get fiber flux at each location

    # fiber_flux_map = convolve(star_img, fiber_allow_small) / star_img.sum()

    # max_point = np.unravel_index(np.argmax(fiber_flux_map), fiber_flux_map.shape)

    # max_value = fiber_flux_map.max()
    # max_point = (max_point[0] - fiber_flux_map.shape[0]/2, max_point[1] - fiber_flux_map.shape[1]/2)

    # print(f"{max_point=}, {max_value=}")

    # y = int(max_point[0])
    # star_line = star_img[y, :]
    # fiber_flux_line = fiber_flux_map[y, :]

    # x_len = star_line.size
    # x2_len = fiber_flux_line.size

    # xs = (np.arange(0, x_len) - x_len/2) * subpixel_len.value
    # x2s = (np.arange(0, x2_len) - x2_len/2) * subpixel_len.value

    # plt.figure()
    # plt.plot(xs, star_line, 'b' )

    # plt.figure()
    # plt.plot(x2s, fiber_flux_line, 'r')
    
    # plt.show()
    # show_img(fiber_flux_map)
    pass

def get_fiber_amount(star_img: np.ndarray, fiber_transmit):
    """Get the percentage of starlight going down the fiber"""
    return (fiber_transmit.value / star_img.sum())

def save_masked_star_img(fname, file_type,
    masked_star_img, subpixel_len, fiber_transmit):
    full_fname = f"{fname}.{file_type}"

    hdu = fits.PrimaryHDU(data=masked_star_img)

    hdu.header['SUBPIXLN'] = subpixel_len.value
    hdu.header['FIBERTX'] = fiber_transmit.value
    hdul = fits.HDUList([hdu])

    # Throws OSError if file already exists, unless overwrite=True
    hdul.writeto(full_fname, overwrite=True)

def load_masked_star_img(fname, file_type):
    full_fname = f"{fname}.{file_type}"

    hdul = fits.open(full_fname)
    hdu = hdul[0]
    subpixel_len = hdu.header['SUBPIXLN'] * u.um
    fiber_transmit = hdu.header['FIBERTX'] * u.ph / u.s
    data = hdu.data

    return data, subpixel_len, fiber_transmit

if __name__ == "__main__":

    plate_scale = 77.34 * u.arcsec / u.mm
    fiber_width = 18 * u.um
    fiber_transmission = 0.95
    fiber_reflection = 0.05

    # Load a star img data
    star_img, subpixel_angle = load_star_img("Test", "fits")

    masked_star_img, subpixel_len, fiber_transmit = apply_fiber_mask(
        star_img, subpixel_angle, plate_scale,
        25, 0, fiber_width, fiber_transmission, fiber_reflection)

    # masked_star_img, subpixel_len, fiber_transmit = load_masked_star_img(
    #     "mask_star_test", "fits"
    # )

    save_masked_star_img("mask_offs_star_test", "fits", masked_star_img,
        subpixel_len, fiber_transmit)

    show_img(masked_star_img, subpixel_len, "photons/sec", "Masked Star")
