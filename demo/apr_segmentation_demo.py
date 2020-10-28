import pyapr
from memory_profiler import memory_usage


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    # Initialize APR and particle objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()  # input particles can be float32 or uint16
    # parts = pyapr.FloatParticles()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    # output must be short (uint16)
    mask = pyapr.ShortParticles()

    # Compute graphcut segmentation
    pyapr.numerics.segmentation.graphcut(apr, parts, mask, 3, 1)

    # Display the result
    pyapr.viewer.parts_viewer(apr, mask)


if __name__ == '__main__':
    max_mem = memory_usage(proc=main, max_usage=True)
    max_mem_mb = max_mem * 1024 ** 2 * 1e-6
    print('Maximum memory usage: {} MiB ({} MB)'.format(max_mem, max_mem_mb))
