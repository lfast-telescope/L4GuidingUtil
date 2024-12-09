import astropy.units as u

from L4_guiding_util.annotated_image import FiberImage, SensorImage
from L4_guiding_util.telescope_sim import capture_image, default_sensor_params

if __name__ == "__main__":
    f = FiberImage.load_fits("test_fiber_img.fits")
    c = capture_image(f, default_sensor_params, 16 * u.ms)

    print(c.snr)

    c.save_png_toml("test_capture_img.png")

    c.save_fits("test_capture_img.fits")
