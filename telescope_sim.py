from dataclasses import dataclass

import numpy as np
import numpy.typing
import astropy.units as u

# TODO: u.pixel_scale equiv, u.plate_scale equiv

@dataclass
class Image(u.Quantity):
    values: u.Quantity
    pixel_scale: tuple()

class Image_Device:
    def __init__(self, forward_value, forward_pixel):
        self.forward_value = forward_value
        self.forward_pixel = forward_pixel

    def forward(self, image, pixel):
        return image * self.forward_value, pixel * self.forward_pixel


LFAST_COLLECTING_AREA = 0.4368 * u.m**2
LFAST_PLATE_SCALE = 77.34 * u.arcsec / u.mm
LFAST_MIRROR_LOSS = 0.95
LFAST_TOP_END_LOSS = 0.98**4

# A description of the lfast telescope mirror and top end, which takes
# A star image where value is 1/collecting area and pixel are u.arcsec
# and goes to an imaging plane image where value is 1 and pixels are u.um

LFAST_FACTOR = 1 / LFAST_PLATE_SCALE**2 * LFAST_MIRROR_LOSS * LFAST_TOP_END_LOSS

lfast_mirror_and_top_end = Image_Device(
    value_to_value=[
        (1/u.arcsec**2, 1, lambda x: x*LFAST_FACTOR,lambda y: y/LFAST_FACTOR,)
    ],
    pixel_to_pixel=[
        (u.arcsec, u.um, 
         lambda x: x/LFAST_PLATE_SCALE, lambda y: y * LFAST_PLATE_SCALE)
    ]
)
