from L4_guiding_util.annotated_image import StarImage
from L4_guiding_util.telescope_sim import (
    apply_telescope,
    apply_fiber,
)


if __name__ == "__main__":

    s = StarImage.load_fits("test_star_img.fits")
    print(f"{s=}")

    apply_telescope(s)

    print(f"{s=}")

    f = apply_fiber(s, (0, 0))

    f.save_fits("test_fiber_img.fits")
