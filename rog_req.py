""" Per Roger Angel's request, this script takes 100 16ms images of polaris from
2024-11-22 on-sky testing of LFAST1x, windows to a specific point, and creates a
histogram of pixel values. Then, these plots are presented one by one.

I used a macro to ctrl+c copy each image, paste into a ppt, ctrl+m to create a
new slide, then close the plot (after which the new plot pops up in the same location)
"""

import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler # must be after matplotlib import. Enables ctrl+c of plot
import astropy.units as u

from sensor_capture_inverse import load_png

from util import show_img_ax



png_fname = "data/16ms_images/2024-11-22-0248_7-CapObj_{:04d}.png"

pixel_scale = 2.4 * u.um / u.pixel
pixel_equiv = u.pixel_scale(pixel_scale)

NUM_FRAMES = 1
START_FRAME = 3
FRAMES_PER_BATCH = 1
window_len = 70
x_center = 172
y_center = 192
x_min_pix = x_center - window_len // 2
x_max_pix = x_min_pix + window_len
y_min_pix = y_center - window_len // 2
y_max_pix = y_min_pix + window_len

window_x = slice(x_min_pix,x_max_pix)
window_y = slice(y_min_pix,y_max_pix)

batches = NUM_FRAMES // FRAMES_PER_BATCH + 1

for batch in range(batches):
    start = START_FRAME+batch*FRAMES_PER_BATCH
    for i in range(start, start+FRAMES_PER_BATCH):
        if i >= START_FRAME + NUM_FRAMES:
            break
        try:
            img = load_png(png_fname.format(i))

        # :exception FileNotFoundError: If the file cannot be found.
        # :exception PIL.UnidentifiedImageError: If the image cannot be opened and
        #    identified.
        # :exception ValueError: If the ``mode`` is not "r", or if a ``StringIO``
        #    instance is used for ``fp``.
        # :exception TypeError: If ``formats`` is not ``None``, a list or a tuple.
        except FileNotFoundError as err:
            print(err)
            continue # continue loop

        # Crop to window
        crop_img = img[window_y, window_x]

        x_min = 0
        y_min = 0
        x_max = (x_max_pix - x_min_pix) * pixel_scale.value
        y_max = (y_max_pix - y_min_pix) * pixel_scale.value

        # For each image, crop to window, create histogram, plot

        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18.0,8.0), gridspec_kw = {"wspace":0.2, "hspace":0.2})
        # # Set absolute location of figure (for macro)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.wm_geometry("+0+0")

        fig.suptitle(f"Frame {i:02d}")
        im = show_img_ax(ax1, crop_img, 
                         pixel_scale*u.pixel, 
                         pixel_scale*u.pixel * 0.07734 * u.arcsec / u.um,
                         f"Image, (Pixel size {pixel_scale})", 
                         imshow_kwargs={"vmin":0.0, "vmax":255.0},)
        img_values = crop_img.flatten()
        ax2.hist(img_values, bins=range(257), log=True)
        ax2.set_title("Histogram of Pixel Values")
        ax2.set_xlabel("Value (8 bit int)")
        ax2.set_ylabel("Number of pixels (log scale)")

    plt.show()