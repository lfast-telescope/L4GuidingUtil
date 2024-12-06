import astropy.units as u

from annotated_image import FiberImage, SensorImage
from telescope_sim import capture_image, default_sensor_params

if __name__ == "__main__":
    f = FiberImage.load_fits("test_fiber_img.fits")
    c = capture_image(f, default_sensor_params, 100 * u.ms)

    c.save_png_toml("test_capture_img.png")

    c.save_fits("test_capture_img.fits")
