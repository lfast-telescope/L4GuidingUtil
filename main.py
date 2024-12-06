
## Generate Test Input
# make fake star (at x/y)
# For now maybe the fake star function just maps an x/y location to an intensity?
# Or we can generate a "real" image... Yeah, real image works
# Apply disk (at x/y)
# Similarly, the disk just passes or fails anything coming through it
# Resample to observed image (Add sampling noise)

# Use a config file for parameters later?
## How to do units in there? Hmmm

import math
import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.modeling.models import Gaussian2D
from astropy.nddata import block_reduce

width = 32
height = 32
scale_up = 8

star_magnitude = 1
star_x_offset = 24.0
star_y_offset = 16.0
star_x_size = 5.0
star_y_size = 5.0

fiber_x_offset = 16.0
fiber_y_offset = 16.0
fiber_radius = 8.0
fiber_transmit = 0.95

# Telescope details needed:
# Telescope aperture = 76.2 cm Use this to get collecting area?
# Mostly plate scale. This can be derived from focal length or
#   from f ratio and aperture
# Aperture might be needed in the star image generation, for an Airy Disk

# Star details needed:
# Magnitude (with respect to zero point)
# Zeropoint?
# Filter
# FWHM full width half maximum?

# Star details bleed into image sensor details.
# Filter or pass band. 
#   Stars have various magnitudes over different pass bands
#   The image sensor has different performance at different pass bands
#   But overall, it seems like we should just be combining all these?
#   Unless we are using the filter wavelength in the Airy disk size...
# So let's just assume a single efficiency?
# Later we can set things by band...

# So step one is to create a star image.
# Choose a scale. Ideally it easily divides into sensor pixel size


# Given:
#   Plate Scale (telescope params)
#   Magnitude (From star params)
#   And FWHM seeing limit
#   Pixel size, dimensions, and upscale factor
# Using star magnitude and flux normalized to a specific filter...
#   Create a model that takes...
#   An x/y imaging plane offset (using plate scale to convert to angle)
#   That x/y offset becomes a point in the Gaussian that gives photons
#       per second per unit area...

# Ideally, we could generate a "real" image or a series of "real" images
# Then we run our image processing on them.
# We want to know how much light goes down the fiber.
# We also want a way to inject images, right?

# We want to:
#   Have an input image or series of images, upscaled
#   Allow us to inject translation noise into these images, frequency controlled
#   Have a model for the L4 translation controls
#   Inject images with L4 translation offsets
#   Apply the fiber transmission
#   Perform Image Processing, apply L4 translation controls

# Exposure times for various update rates:
# 60 fps: 16.667 ms
# 45 fps: 22.222 ms
# 30 fps: 33.333 ms
# 22 fps: 45.454 ms
# 15 fps: 66.667 ms
# 10 fps: 100.00 ms

# standard photometric values for various filter:
# central wavelength, filter width, flux zeropoint, typical dark sky background (arxiv Pedani2009 Mt Graham )
ZP = {
    "U": [0.360*u.um,  0.050*u.um, 4.190e-9*u.erg/u.s/u.cm**2/u.angstrom, 21.98*u.mag/u.arcsec**2],
    "B": [0.440*u.um,  0.072*u.um, 6.320e-9*u.erg/u.s/u.cm**2/u.angstrom, 22.81*u.mag/u.arcsec**2],
    "V": [0.550*u.um,  0.086*u.um, 3.630e-9*u.erg/u.s/u.cm**2/u.angstrom, 21.81*u.mag/u.arcsec**2],
    "R": [0.710*u.um,  0.133*u.um, 2.177e-9*u.erg/u.s/u.cm**2/u.angstrom, 20.82*u.mag/u.arcsec**2],
    "I": [0.970*u.um,  0.140*u.um, 1.126e-9*u.erg/u.s/u.cm**2/u.angstrom, 19.78*u.mag/u.arcsec**2],
    "L": [0.600*u.um,  0.300*u.um, 3.000e-9*u.erg/u.s/u.cm**2/u.angstrom, 20.84*u.mag/u.arcsec**2]
}
def getstarflux(mag=0, filter="V"):
    # return the number of photons per second received above the atmosphere for a given filter, magnitude, and collecting area.
    psm = ZP[filter][2].to(u.ph/u.s/u.m**2/u.um, u.spectral_density(ZP[filter][0]))*ZP[filter][1]
    photon_flux = psm*10**(-mag/2.5) # photons per second per unit area
    return ps


# Need to know... amount to upscale the star?
def make_fake_star():
    """Creates an upscaled image of a Gaussian star"""
    star_model = Gaussian2D(1.0, 
        star_x_offset * scale_up, 
        star_y_offset * scale_up,
        star_x_size * scale_up,
        star_y_size * scale_up)
    x,y = np.meshgrid(np.arange(0,width * scale_up), np.arange(0,height*scale_up))
    star_img = star_model(x,y)
    # Normalize flux to stellar magnitude
    star_img = star_img/star_img.sum() * star_magnitude
    return star_img

    
def create_fiber_mask():
    """Creates a mask of upscaled size that describes the image lost to the fiber"""

    fiber_input = np.zeros((width * scale_up, height * scale_up))
    for i in range(width * scale_up):
        for j in range(height * scale_up):
            x = i / scale_up
            y = j / scale_up
            if math.sqrt((x - fiber_x_offset)**2 + (y-fiber_y_offset)**2) < fiber_radius:
                fiber_input[i,j] = fiber_transmit

    fiber_mask = np.ones(fiber_input.shape) - fiber_input

    return fiber_mask, fiber_input



def resample_observe(star_img, fiber_mask, fiber_input):
    """Given a fake star and a fiber mask, resamples to observed image
    and to an amount of flux down the fiber"""

    masked_star_img = star_img * fiber_mask

    # Resample
    observed_img = block_reduce(masked_star_img, scale_up)

    fiber_flux = (star_img * fiber_input).sum()

    return observed_img, fiber_flux



def generate_small_test_input():
    # Make fake star

    star_img = make_fake_star()

    # Make fiber mask

    fiber_mask, fiber_input = create_fiber_mask()

    # Resample/Observe "real" image

    observed_img, _ = resample_observe(star_img, fiber_mask, fiber_input)

    return observed_img

def get_centroid(data):
    total = np.sum(data)

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]

    # note the output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total
                     for axis in range(data.ndim)])[::-1]

if __name__ == "__main__":

    obs = generate_small_test_input()
    plt.imshow(obs, cmap='grey')
    cx, cy = get_centroid(obs)
    plt.plot(cx, cy, "ro")
    plt.plot(fiber_x_offset - 0.5,fiber_y_offset - 0.5,'go')
    plt.plot(star_x_offset - 0.5,star_y_offset - 0.5,'bo')
    plt.show()

## Do image processing

