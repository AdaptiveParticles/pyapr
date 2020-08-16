import pyapr
import numpy as np
from skimage import io as skio


def main():

    # Get input image file path from gui
    io_int = pyapr.filegui.InteractiveIO()
    fpath = io_int.get_tiff_file_name()

    # Read image and convert to float
    img = skio.imread(fpath).astype(np.float32)

    # Add a small offset to avoid division by 0
    img += 1e-5 * img.max()

    # Specify the PSF and number of iterations
    psf = np.ones((5, 5, 5), dtype=np.float32) / 125
    num_iter = 10

    # Perform richardson-lucy deconvolution
    output = np.empty(img.shape, dtype=np.float32)
    pyapr.numerics.richardson_lucy_pixel_cuda(img, output, psf, num_iter)  # currently no CPU version available

    # Get save file path from gui and write the result
    save_path = io_int.save_tiff_file_name()
    skio.imsave(save_path, output)


if __name__ == '__main__':
    main()
