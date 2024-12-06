import numpy as np
from matplotlib import pyplot as plt
import astropy
from astropy.modeling.models import Gaussian2D
import astropy.units as u
import astropy.io.fits as fits
import scipy.interpolate

from annotated_image import *

import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

if __name__ == "__main__":
    logging.basicConfig()

# TODO: figure out how to make this process deterministic?
noise_rng = np.random.default_rng()

lfast1x_telescope_params = TelescopeParams(
    plate_scale=0.07734 * u.arcsec / u.um,
    aperture=0.762 * u.m,
    collecting_area=0.4368 * u.m**2,
    forward_losses=0.80 * (0.98**8),  # Primary mirror + 8 lens surfaces
)


lfast1x_relay_params = RelayParams(
    forward_losses=(0.98**8),  # 8 lens surfaces
)


lfast1x_fiber_params = FiberParams(
    width=18 * u.um, transmit=0.98, reflect=0.01, mirror_reflect=0.90
)

default_sensor_params = SensorParams(
    name="default",
    pixel_pitch=1.85 * u.um / u.pix,
    read_noise=3.0 * u.electron,
    gain=0.35 * u.adu / u.electron,
    dark_current=0.26 * u.electron / u.s,
    bias=0 * u.adu,
    quantum_efficiency=0.60 * u.electron / u.photon,
)


def get_zero_point_flux(filter="V"):
    return (
        (3.630e-9 * u.erg / u.s / u.cm**2 / u.angstrom).to(
            u.ph / u.s / u.m**2 / u.um, u.spectral_density(0.550 * u.um)
        )
        * 0.086
        * u.um
    )


# fmt: off
# (central wavelength, filter width, flux zeropoint, typical dark sky background)
ZP: dict[str, tuple[u.Quantity[u.um], u.Quantity[u.um], u.Quantity, u.Quantity]] = {
    "U": (0.360*u.um,  0.050*u.um, 4.190e-9*u.erg/u.s/u.cm**2/u.angstrom, 21.98*u.mag/u.arcsec**2),
    "V": (0.550*u.um,  0.086*u.um, 3.630e-9*u.erg/u.s/u.cm**2/u.angstrom, 21.81*u.mag/u.arcsec**2),
    "B": (0.440*u.um,  0.072*u.um, 6.320e-9*u.erg/u.s/u.cm**2/u.angstrom, 22.81*u.mag/u.arcsec**2),
    "R": (0.710*u.um,  0.133*u.um, 2.177e-9*u.erg/u.s/u.cm**2/u.angstrom, 20.82*u.mag/u.arcsec**2),
    "I": (0.970*u.um,  0.140*u.um, 1.126e-9*u.erg/u.s/u.cm**2/u.angstrom, 19.78*u.mag/u.arcsec**2),
    "L": (0.600*u.um,  0.300*u.um, 3.000e-9*u.erg/u.s/u.cm**2/u.angstrom, 20.84*u.mag/u.arcsec**2)
}
# fmt: on


def total_star_flux(mag: float, filter="L"):
    """Number of (photons per second per collecting area) received above the atmosphere for a given filter, and magnitude"""
    wav, width, zp, _ = ZP[filter]  # wavelength, width of band, zeropoint flux
    photon_flux_units = u.ph / u.s / u.m**2 / u.um
    filter_zp_flux = zp.to(photon_flux_units, u.spectral_density(wav)) * width
    return 10 ** (-mag / 2.5) * filter_zp_flux


def generate_star_gaussian(
    shape,
    fwhm: u.Quantity["angle"] = None,
    pixel_scale: u.Quantity = None,  # units/pixel
    magnitude: float = None,
) -> StarImage:
    # Gaussian standard deviation, in pixels
    sigma = fwhm / 2.355 / pixel_scale

    star_model = Gaussian2D(1.0, 0.0, 0.0, sigma.value, sigma.value)

    # Generate an x and y range over the region, in pixels?
    x_range = np.arange(shape[1]) - shape[1] / 2
    y_range = np.arange(shape[0]) - shape[0] / 2

    x, y = np.meshgrid(x_range, y_range)

    star_img = StarImage(star_model(x, y), pix_sca=pixel_scale, normaled=False)

    if magnitude != None:
        normalize_star(star_img, magnitude)

    return star_img


def normalize_star(star_img: StarImage, magnitude: float):
    """Normalizes star image to a magnitude"""
    total_flux = total_star_flux(magnitude)
    scalar = 1 / star_img.values.sum() * total_flux
    star_img.apply_scalar(scalar)
    star_img.normaled = True


def apply_telescope(
    star_img: StarImage,
    params: TelescopeParams = lfast1x_telescope_params,
    inverse=False,
):
    """Changes StarImage from arcsec scale to um scale, accounting for losses"""
    op = lambda x: x if not inverse else lambda x: 1 / x

    star_img.apply_scalar(op(params.forward_losses))
    star_img.apply_scalar(op(params.collecting_area))

    unit_to, unit_from = (u.um, u.arcsec) if not inverse else (u.arcsec, u.um)
    if star_img.pix_sca.unit != unit_from / u.pix:
        raise ValueError(
            f"Applying telescope to img {star_img}, got unexpected pix_sca unit: {star_img.pix_sca.unit}, expected {unit_from / u.pix}"
        )

    old_pix_sca = star_img.pix_sca
    plate_scale = u.plate_scale(params.plate_scale)
    star_img.pix_sca = (star_img.pix_sca * u.pix).to(unit_to, plate_scale) / u.pix
    if old_pix_sca != star_img.pix_sca:
        star_img.pix_sca2 = old_pix_sca


def apply_relay(
    fiber_img: FiberImage, params: RelayParams = lfast1x_relay_params, inverse=False
):
    op = lambda x: x if not inverse else lambda x: 1 / x
    fiber_img.apply_scalar(op(params.forward_losses))


def create_disk_map(shape: tuple[int, int], radius: float, inside_value, outside_value):
    # Generate an x and y range over the region, in pixels
    x_range = np.arange(shape[1]) - shape[1] / 2
    y_range = np.arange(shape[0]) - shape[0] / 2

    x, y = np.meshgrid(x_range, y_range)

    disk = x * x + y * y <= radius**2

    disk_map = np.ones(shape) * outside_value
    disk_map[disk] = inside_value

    return disk_map


def apply_fiber(
    star_img: StarImage, star_offsets, params: FiberParams = lfast1x_fiber_params
) -> FiberImage:

    shift = translate_img(star_img.values, star_offsets[0], star_offsets[1])

    # Create fiber transmit and fiber reflect maps
    radius = params.width.to(u.pix, star_img.ps_eq()).value / 2
    fiber_transmit = create_disk_map(shift.shape, radius, params.transmit, 0.0)
    fiber_reflect = create_disk_map(
        shift.shape, radius, params.reflect, params.mirror_reflect
    )
    fiber_transmit_map = shift * fiber_transmit * star_img.unit
    fiber_reflect_map = shift * fiber_reflect * star_img.unit

    # Sum encircled energy
    fiber_flux = fiber_transmit_map.sum()

    # Create and return Fiber image
    fiber_image = FiberImage(
        fiber_reflect_map,
        fib_flux=fiber_flux,
        pix_sca=star_img.pix_sca,
        pix_sca2=star_img.pix_sca2,
    )
    return fiber_image


def reimage(img: AnnotatedImage, new_pix_sca: u.Quantity, method="linear"):
    # New shape will be based on relative scale...
    old_shape = img.value.shape
    width = old_shape[1] * img.pix_sca.value
    height = old_shape[0] * img.pix_sca.value
    shape = (int(height / new_pix_sca.value), int(width / new_pix_sca.value))

    fit_points = [
        np.linspace(0, height, old_shape[0]),
        np.linspace(0, width, old_shape[1]),
    ]
    x, y = np.meshgrid(
        np.linspace(0, height, shape[0]), np.linspace(0, width, shape[0])
    )
    test_points = np.array([x.ravel(), y.ravel()]).T

    interp = scipy.interpolate.RegularGridInterpolator(fit_points, img.value)
    reimg = interp(test_points, method).reshape(shape[1], shape[0])
    return img.new_from_value(reimg)


def capture_image(
    fiber_img: FiberImage,
    params: SensorParams = default_sensor_params,
    exposure: u.Quantity["time"] = 16 * u.ms,
):
    """Simulate capturing an image with the given sensor
    If pixel scale is not appropriate for a block_reduce, interpolate to appropriate size
    """
    # Get scale of fiber img pixels to image sensor pixels
    # fiber img u.um / pix
    scale = (params.pixel_pitch / fiber_img.pix_sca).value
    if abs(int(scale + 0.5) - scale) > 0.10:
        # If within 1%, proceeed, otherwise, interpolate
        logging.warning(f"{scale=}, bad scale")
        logger
        scale = int(np.ceil(scale))
        desired_pitch = params.pixel_pitch * scale
        fiber_img = reimage(fiber_img, desired_pitch)

    scale = int(scale + 0.5)  # Round to int
    # Bin photons to pixels
    signal_img = astropy.nddata.block_reduce(fiber_img.values, scale) * fiber_img.unit
    pix_sca2 = fiber_img.pix_sca2 * scale
    shape = signal_img.shape
    # Quantize data
    noise_counts = (params.read_noise * params.gain).to(u.adu).value

    # TODO: make noise seeded from image details?
    # Read Noise is a normal distribution from the bias
    noise_img = noise_rng.normal(params.bias.value, noise_counts, size=shape) * u.adu
    # Dark current is a poisson distribution as it is an expected mean rate
    dc_rate = (params.dark_current * exposure * params.gain).to(u.adu)
    dark_img = noise_rng.poisson(dc_rate.value, size=shape) * u.adu

    # Signal is photons to electrons to counts
    signal_img *= exposure * params.quantum_efficiency * params.gain

    try:
        quantized_img = (noise_img + dark_img + signal_img).value
    except u.core.UnitConversionError as uce:
        logger.error(f"Failed to add imgs: {noise_img=}, {dark_img=}, {signal_img=}")
        raise uce

    quantized_img = np.clip(quantized_img, 0, 2**16 - 1)

    sensor_img = SensorImage(
        quantized_img,
        pix_sca=params.pixel_pitch,
        pix_sca2=pix_sca2,
        exposure=exposure,
        sensor_params=params,
    )

    return sensor_img
