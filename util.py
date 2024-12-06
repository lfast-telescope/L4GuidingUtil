import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import addcopyfighandler

import astropy.modeling
import astropy.units as u
import astropy.io.fits as fits

def show_img_ax(ax: matplotlib.axes.Axes, img_data: np.ndarray, xy_unit_scale = None, xy2_unit_scale = None, title = None, origin="center", imshow_kwargs=None):
    if imshow_kwargs == None:
        imshow_kwargs = {}
    if xy_unit_scale == None:
        xy_unit_scale = 1.0 * u.pix

    if xy2_unit_scale is not None:
        scale2 = (xy2_unit_scale / xy_unit_scale).value
    else:
        scale2 = 1 # Not used

    region_shape = img_data.shape

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

    # Plot image in grayscale, with axes in arcsec
    im = ax.imshow(
        img_data,
        cmap = 'gray',
        extent = (x_min, x_max, y_min, y_max),
        **imshow_kwargs
    )

    if xy2_unit_scale is not None:
        y2 = ax.secondary_yaxis('right', functions=(lambda x: x*scale2, lambda y: y/scale2))
        y2.set_ylabel(str(xy2_unit_scale.unit))
        x2 = ax.secondary_xaxis('top', functions=(lambda x: x*scale2, lambda y: y/scale2))
        x2.set_xlabel(str(xy2_unit_scale.unit))



    xy_unit_name = str(xy_unit_scale.unit)
    if not title == None:
        ax.set_title(str(title))
    ax.set_xlabel(xy_unit_name)
    ax.set_ylabel(xy_unit_name)

    return im

def show_img(img_data: np.ndarray, xy_unit_scale = None, data_unit = None, title = None, origin ="center"):

    fig, ax = plt.subplots()

    if title is not None:
        fig.suptitle(title)

    im = show_img_ax(ax, img_data, xy_unit_scale, None, origin)

    if not data_unit == None:
        fig.colorbar(im, ax=ax, label=str(data_unit))

def translate_img(img: np.ndarray, offset_x, offset_y) -> np.ndarray:
    x = -int(offset_x)
    y = -int(offset_y)

    x_len = img.shape[1]
    y_len= img.shape[0]

    src_slice_x =  slice(max(0,x),  min(x_len,x_len+x))
    src_slice_y =  slice(max(0,y),  min(y_len,y_len+y))
    dest_slice_x = slice(max(-x,0), min(x_len,x_len-x))
    dest_slice_y = slice(max(-y,0), min(y_len,y_len-y))

    shift_img = np.zeros((y_len, x_len))

    shift_img[dest_slice_y,dest_slice_x] = img[src_slice_y, src_slice_x]

    return shift_img