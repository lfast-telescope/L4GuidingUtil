import numpy as np

def get_centroid(img_data):
    # Threshold data to isolate centroid calculation to star
    min_pix = np.min(img_data)
    max_pix = np.max(img_data)

    d_pix = max_pix - min_pix
    thresh = min_pix + int(d_pix * .1)

    img_data = np.where(img_data < thresh, 0, img_data)

    x,y = np.meshgrid(
        np.arange(0, img_data.shape[1]),
        np.arange(0, img_data.shape[0])
    )

    x_center = np.sum(img_data * x) / img_data.sum()
    y_center = np.sum(img_data * y) / img_data.sum()

    return x_center, y_center


