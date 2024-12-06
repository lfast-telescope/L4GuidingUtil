import numpy as np
from matplotlib import pyplot as plt
import astropy
import astropy.units as u
import astropy.io.fits as fits

from fiber_mask_application import load_masked_star_img
from util import show_img

test_sensor_params = {
    'pixel_len':1.85 * u.um,
    'read_noise':3.0 * u.electron,
    'gain': 0.35 * u.adu / u.electron,
    'dark_current': 0.26 * u.electron / u.s,
    'bias': 1000 * u.adu,
    'quantum_efficiency': 0.60,
}

# TODO: figure out how to make this process deterministic?
noise_rng = np.random.default_rng()

def bin_photons_to_pixels(
    img_data, subpixel_len, pixel_len):

    scale_float = (pixel_len / subpixel_len).value

    # Check that scale is close to an integer
    scale = int(scale_float + 0.5)
    if abs(scale-scale_float) / scale_float > 0.1:
        raise ValueError("subpixel - pixel scale is not an integer value")

    desired_shape_x = img_data.shape[1] // scale * scale
    desired_shape_y = img_data.shape[0] // scale * scale
    offset_x = (img_data.shape[1] - desired_shape_x)//2
    offset_y = (img_data.shape[0] - desired_shape_y)//2

    desired_img = img_data[
        offset_y:offset_y + desired_shape_y,
        offset_x:offset_x + desired_shape_x
    ]

    scaled_img = astropy.nddata.block_reduce(desired_img, scale)

    return scaled_img

def quantize_sensor_img(
    img_data, exposure, im_sensor_param):
    """
    exposure: time, u.s or u.ms
    im_sensor_param contains:
    read_noise: u.electron deviation from mean
    gain: u.adu / u.electron counts per electron
    quantum_efficiency: float (represents u.electron / u.photon)
    dark_current: u.electron / u.s
    bias: u.adu (offset counts)
    """
    read_noise = im_sensor_param['read_noise'] # u.electron
    gain = im_sensor_param['gain'] # u.adu / u.electron
    dark_current = im_sensor_param['dark_current'] # u.electron / u.s
    bias = im_sensor_param['bias'] # u.adu
    quantum_efficiency = im_sensor_param['quantum_efficiency'] # float
    shape = img_data.shape

    # Read Noise is a normal distribution from the bias
    noise_counts = (read_noise*gain).to(u.adu).value
    noise_img = noise_rng.normal(bias.value, noise_counts, size=shape)

    # Dark current is a poisson distribution as it is an expected mean rate
    dark_current_rate = (dark_current*exposure*gain).to(u.adu).value
    dark_img = noise_rng.poisson(dark_current_rate, size=shape)
    
    # Signal is photons to electrons to counts
    signal_img = img_data*exposure.value*quantum_efficiency*(gain.value)

    quantized_img = noise_img + dark_img + signal_img
    quantized_img = np.clip(quantized_img, 0, 2**16 - 1)
    return quantized_img


def simulate_sensor_capture(
    img_data, subpixel_len, exposure, im_sensor_param):

    pixel_len = im_sensor_param['pixel_len']
    sensor_img = bin_photons_to_pixels(
        img_data, subpixel_len, pixel_len)

    # Quantize data to counts, including noise and bias

    quantized_img = quantize_sensor_img(
        sensor_img, exposure, im_sensor_param)

    return quantized_img

if __name__ == "__main__":

    im_sensor_param = test_sensor_params

    masked_star_img, subpixel_len, _ = \
        load_masked_star_img("mask_offs_star_test","fits")

    final_img = simulate_sensor_capture(
        masked_star_img, subpixel_len, 16*u.ms, im_sensor_param)

    show_img(final_img, None, "counts", "Image Sensor")


    