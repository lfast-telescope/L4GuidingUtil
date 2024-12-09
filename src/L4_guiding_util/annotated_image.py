import numpy as np
import numpy.typing as npt
import astropy.units as u
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import json
import png
from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

# NOTE: Use *= operator to apply a scalar to an image

__all__ = [
    "TelescopeParams",
    "FiberParams",
    "RelayParams",
    "SensorParams",
    "AnnotatedImage",
    "StarImage",
    "FiberImage",
    "SensorImage",
    "translate_img",
]


def all_subclasses(cls):
    """Return all subclasses of cls"""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


# Note: Could later expand this based on wavelength
@dataclass
class TelescopeParams:
    plate_scale: u.Quantity[u.arcsec / u.um]
    aperture: u.Quantity[u.m]
    collecting_area: u.Quantity
    forward_losses: float


@dataclass
class FiberParams:
    width: u.Quantity["length"]
    transmit: float
    reflect: float
    mirror_reflect: float


@dataclass
class RelayParams:
    forward_losses: float


@dataclass
class SensorParams:
    name: str
    pixel_pitch: u.Quantity["length"]
    read_noise: u.Quantity[u.electron]
    gain: u.Quantity[u.adu / u.electron]
    dark_current: u.Quantity[u.electron / u.s]
    bias: u.Quantity[u.adu]
    quantum_efficiency: u.Quantity[u.electron / u.photon]

    def to_json(self):
        d = self.__dict__.copy()
        for k in d:
            if isinstance(d[k], u.Quantity):
                d[k] = str(d[k])
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str):
        return cls(**(json.loads(json_str)))

    def to_toml(self):
        s = f"[{self.__class__.__name__}]\n"
        for k, v in self.__dict__.items():
            if isinstance(v, u.Quantity):
                v = str(u.Quantity)
            s += f"[{k}='{v}']\n"
        return s


# TODO: make this more generic... Ok to not use quantity as is!
class AnnotatedImage:
    """Wrapper for u.Quantity (which itself wraps np.ndarray)
    Also has a pixel scale annotation
    Can still be used as a Quantity (i.e. .value to get array, mult/divide units)
    annotation names should be convertible to FITS header keywords with .upper()
    annotation values should be either a basic fits type, or convertible to a string and back
    """

    def __init__(self, values, unit=None, pix_sca=None, pix_sca2=None, **kwargs):
        if unit is None:
            unit = u.dimensionless_unscaled
        if isinstance(values, u.Quantity):
            self.values = values.value
            self.unit = values.unit * unit
        else:
            self.values = values
            self.unit = unit
        self.pix_sca = pix_sca
        self.pix_sca2 = pix_sca2

    def quantity(self):
        return self.values * self.unit

    def apply_scalar(self, scalar):
        if isinstance(scalar, u.Quantity):
            self.unit *= scalar.unit
            scalar = scalar.value
        self.values *= scalar

    @classmethod
    def from_hdu(cls, hdu):
        values = hdu.data
        unit = u.Unit(hdu.header["UNIT"])
        pix_sca = u.Quantity(hdu.header["pix_sca"])
        pix_sca2 = None
        if "pix_sca2" in hdu.header:
            pix_sca2 = u.Quantity(hdu.header["pix_sca2"])
        return cls(values, unit=unit, pix_sca=pix_sca, pix_sca2=pix_sca2)

    @classmethod
    def load_fits(cls, fname):
        logger.debug(f"Loading fits file {fname}, expecting class {cls}")
        hdul = fits.open(fname)
        hdu = hdul[0]  # primary hdu
        return cls.load_hdu(hdu, fname)

    @classmethod
    def load_hdu(cls, hdu, fname=None):
        class_name = hdu.header["CLASSNM"]
        if class_name != cls.__name__:
            raise TypeError(
                f"File {fname} header 'CLASSNM' contained wrong class "
                f"name. Got {class_name} but expected {cls.__name__}"
            )
        return cls.from_hdu(hdu)

    def write_hdu(self, hdu):
        hdu.header["UNIT"] = str(self.unit)
        hdu.header["CLASSNM"] = self.__class__.__name__
        hdu.header["pix_sca"] = str(self.pix_sca)
        if self.pix_sca2 is not None:
            hdu.header["pix_sca2"] = str(self.pix_sca2)
        self.write_hdu_ctype(hdu)

    def write_hdu_ctype(self, hdu):
        unit = (self.pix_sca * u.pix).unit
        value = self.pix_sca.value
        hdu.header["CTYPE1"] = str(unit)
        hdu.header["CTYPE2"] = str(unit)
        hdu.header["CDELT1"] = value
        hdu.header["CDELT2"] = value
        # Get center of image
        shape = self.values.shape
        xc = (shape[0] + 1) / 2  # 1 based indexing...
        yc = (shape[1] + 1) / 2
        hdu.header["CRPIX1"] = xc
        hdu.header["CRVAL1"] = 0.0
        hdu.header["CRPIX2"] = yc
        hdu.header["CRVAL2"] = 0.0

    def save_fits(self, fname, overwrite=True):
        hdu = fits.PrimaryHDU(data=self.values)
        self.write_hdu(hdu)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=overwrite)

    def ps_eq(self) -> u.Equivalency:
        return u.pixel_scale(self.pix_sca)

    def __repr__(self):
        repr = (
            f"<{self.__class__.__name__} {self.values.shape} array of "
            f"{self.unit}, pix_sca={self.pix_sca}>"
        )
        return repr

    @staticmethod
    def generic_load_fits(fname):
        subclasses = set.union(all_subclasses(AnnotatedImage), {AnnotatedImage})
        logger.debug(f"Loading fits file {fname}, expecting one of {subclasses}")
        hdul = fits.open(fname)
        hdu = hdul[0]  # primary hdu
        class_name = hdu.header["CLASSNM"]
        for subcls in subclasses:
            if subcls.__name__ == class_name:
                return subcls.load_hdu(hdu, fname)

    def show_img(self, title=None, origin="center"):
        fig, ax = plt.subplots()

        if title is not None:
            fig.suptitle(title)
        im = self.show_img_ax(ax, origin)

    def show_img_ax(self, ax, origin="center"):
        xy_unit_scale = self.pix_sca
        if xy_unit_scale is None:
            xy_unit_scale = 1.0 * u.pix
        xy_unit_scale2 = self.pix_sca2

        region_shape = self.values.shape

        if origin == "center":
            x_min = -region_shape[1] / 2
            x_max = x_min + region_shape[1]
            x_min *= xy_unit_scale.value
            x_max *= xy_unit_scale.value

            y_min = -region_shape[0] / 2
            y_max = y_min + region_shape[0]
            y_min *= xy_unit_scale.value
            y_max *= xy_unit_scale.value

        if origin == "corner":
            x_min = 0
            x_max = region_shape[1] * xy_unit_scale.value
            y_min = region_shape[0] * xy_unit_scale.value
            y_max = 0

        im = ax.imshow(self.values, cmap="gray", extent=(x_min, x_max, y_min, y_max))

        if xy_unit_scale2 is not None:
            scale2 = (xy_unit_scale2 / xy_unit_scale).value
            y2 = ax.secondary_yaxis(
                "right", functions=(lambda x: x * scale2, lambda y: y / scale2)
            )
            y2.set_ylabel(str(xy_unit_scale2.unit))
            x2 = ax.secondary_xaxis(
                "top", functions=(lambda x: x * scale2, lambda y: y / scale2)
            )
            x2.set_xlabel(str(xy_unit_scale2.unit))

        xy_unit_name = str(xy_unit_scale.unit)
        ax.set_xlabel(xy_unit_name)
        ax.set_ylabel(xy_unit_name)
        return im


class StarImage(AnnotatedImage):
    """An image of a star, usually with a unit of photons/sec/area or
    photons/sec"""

    def __init__(self, value, normaled: bool = False, **kwargs):
        super().__init__(value, **kwargs)
        self.normaled = normaled

    @classmethod
    def from_hdu(self, hdu):
        s = super().from_hdu(hdu)
        s.normaled = hdu.header["normaled"]
        return s

    def write_hdu(self, hdu):
        super().write_hdu(hdu)
        hdu.header["normaled"] = self.normaled

    def __repr__(self):
        s = super().__repr__()
        s = s[0:-1] + f", normaled={self.normaled}" + s[-1:]
        return s


class FiberImage(AnnotatedImage):
    """An image of a star, usually with a unit of photons/sec, pixel scale in um
    Also has a fib_flux annotation which is the amount of flux down the fiber
    """

    def __init__(self, value, fib_flux: u.Quantity = None, **kwargs):
        super().__init__(value, **kwargs)
        self.fib_flux = fib_flux

    @classmethod
    def from_hdu(self, hdu):
        s = super().from_hdu(hdu)
        s.fib_flux = u.Quantity(hdu.header["fib_flux"])
        return s

    def write_hdu(self, hdu):
        super().write_hdu(hdu)
        hdu.header["fib_flux"] = str(self.fib_flux)

    def fib_flux_ratio(self) -> float:
        """Get the ratio of light in image that went down fiber"""
        img_flux: u.Quantity = self.quantity().sum()
        fib_flux: u.Quantity = self.fib_flux
        if img_flux.unit != fib_flux.unit:
            try:
                fib_flux = fib_flux.to(img_flux.unit)
            except Exception:
                raise TypeError(
                    f"{self.__class__.__name__} image flux ({img_flux}) "
                    f"cannot convert to fiber flux ({fib_flux})"
                )
        return (fib_flux / (img_flux + fib_flux)).value

    def __repr__(self):
        s = super().__repr__()
        s = s[0:-1] + f", fib_flux={self.fib_flux}" + s[-1:]
        return s


class SensorImage(AnnotatedImage):
    """Image quantized to a sensor"""

    def __init__(
        self,
        values,
        exposure: u.Quantity = None,
        sensor_params: SensorParams = None,
        **kwargs,
    ):
        super().__init__(values, **kwargs)
        self.values = np.rint(self.values).astype(np.dtype(np.uint16))
        if isinstance(exposure, u.Quantity):
            exposure = exposure.to(u.s).value
        self.exposure = exposure
        self.sensor_params = sensor_params
        if "pix_sca" not in kwargs and self.sensor_params is not None:
            self.pix_sca = self.sensor_params.pixel_pitch

    @classmethod
    def from_hdu(self, hdu):
        s = super().from_hdu(hdu)
        if "exposure" in hdu.header:
            self.exposure = hdu.header["exposure"]
        if "SNSRJSON" in hdu.header:
            self.sensor_params = SensorParams.from_json(hdu.header["SNSRJSON"])
        return s

    def write_hdu(self, hdu):
        super().write_hdu(hdu)
        if self.exposure is not None:
            hdu.header["exposure"] = self.exposure
        if self.sensor_params is not None:
            hdu.header["SNSRJSON"] = self.sensor_params.to_json()

    def __repr__(self):
        s = super().__repr__()
        s = s[0:-1] + f", exposure={self.exposure}" + s[-1:]
        s = s[0:-1] + f", sensor_params={self.sensor_params}" + s[-1:]
        return s

    def save_png_toml(self, png_fname: str, overwrite=True):
        """Save to a png file with a .txt file containing annotation info"""
        # Create an Image with format 32 bit integers
        data = self.values
        rows = [data[i, :] for i in range(data.shape[0])]
        png_w = png.Writer(
            width=data.shape[1],
            height=data.shape[0],
            bitdepth=16,
            greyscale=True,
        )
        with open(png_fname, "wb") as f:
            png_w.write(f, rows)

        toml_fname = png_fname.replace(".png", ".toml")
        with open(toml_fname, "w") as f:
            for k, v in self.__dict__.items():
                if k in ["values", "sensor_params"]:
                    continue
                if v is None:
                    continue
                f.write(f"{k}='{v}'")
            if self.sensor_params is not None:
                f.write(self.sensor_params.to_toml())


def translate_img(img: np.ndarray, offset_x, offset_y) -> u.Quantity:
    """Backfill with zeroes"""
    x = -int(offset_x)
    y = -int(offset_y)

    x_len = img.shape[1]
    y_len = img.shape[0]

    src_slice_x = slice(max(0, x), min(x_len, x_len + x))
    src_slice_y = slice(max(0, y), min(y_len, y_len + y))
    dest_slice_x = slice(max(-x, 0), min(x_len, x_len - x))
    dest_slice_y = slice(max(-y, 0), min(y_len, y_len - y))

    shift_img = np.zeros((y_len, x_len))

    shift_img[dest_slice_y, dest_slice_x] = img[src_slice_y, src_slice_x]
    return shift_img
