import pyapr
import numpy as np
from time import time


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    # Initialize APR and particle objects
    parts = pyapr.ShortParticles()
    apr = pyapr.APR()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    # Illustrates the usage of the Python-wrapped linear iterator by computing the piecewise constant reconstruction
    start = time()
    org_dims = apr.org_dims()  # dimension order (y, x, z)
    py_recon = np.empty((org_dims[2], org_dims[1], org_dims[0]), dtype=np.uint16)
    max_level = apr.level_max()

    apr_it = apr.iterator()  # wrapper of linear iterator

    # loop over levels up to level_max-1
    for level in range(apr_it.level_min(), apr_it.level_max()):

        step_size = 2 ** (max_level - level)

        for z in range(apr_it.z_num(level)):
            for x in range(apr_it.x_num(level)):
                for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                    y = apr_it.y(idx)  # this is slow

                    y_start = y * step_size
                    x_start = x * step_size
                    z_start = z * step_size

                    y_end = min(y_start+step_size, py_recon.shape[2])
                    x_end = min(x_start+step_size, py_recon.shape[1])
                    z_end = min(z_start+step_size, py_recon.shape[0])

                    py_recon[z_start:z_end, x_start:x_end, y_start:y_end] = parts[idx]

    # particles at the maximum level coincide with pixels
    level = max_level
    for z in range(apr_it.z_num(level)):
        for x in range(apr_it.x_num(level)):
            for idx in range(apr_it.begin(level, z, x), apr_it.end()):
                py_recon[z, x, apr_it.y(idx)] = parts[idx]

    py_time = time()-start
    print('python reconstruction took {} seconds'.format(py_time))

    # Compare to the c++ reconstruction
    start = time()
    tmp = pyapr.numerics.reconstruction.recon_pc(apr, parts)
    cpp_recon = np.array(tmp, copy=False)
    cpp_time = time()-start
    print('c++ reconstruction took {} seconds'.format(cpp_time))
    print('c++ was {} times faster'.format(py_time / cpp_time))

    # check that both methods produce the same results (on a subset of the image if it is larger than 128^3 pixels)
    zm = min(org_dims[2], 128)
    xm = min(org_dims[1], 128)
    ym = min(org_dims[0], 128)

    success = np.allclose(py_recon[:zm, :xm, :ym], cpp_recon[:zm, :xm, :ym])
    if not success:
        print('Python and C++ reconstructions seem to give different results...')


if __name__ == '__main__':
    main()