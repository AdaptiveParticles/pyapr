import pyapr
import numpy as np
from skimage import io as skio


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()

    img = skio.imread(fpath).astype(np.float32)
    img += 1e-5 * img.max()  # add a small offset to avoid division by 0

    psf = np.ones((5, 5, 5), dtype=np.float32) / 125
    num_iter = 20

    output = np.empty(img.shape, dtype=np.float32)
    pyapr.numerics.richardson_lucy_pixel_cuda(img, output, psf, num_iter)

    save_path = io_int.save_tiff_file_name()
    skio.imsave(save_path, output)

    # do something with output


if __name__ == '__main__':
    main()
