import os
import pyapr


def main():

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()  # get APR file path from gui

    # Initialize objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()

    # Read APR and particles from file
    pyapr.io.read(fpath_apr, apr, parts)

    # Interactive WNL compression
    pyapr.viewer.interactive_compression(apr, parts)

    # Write compressed APR to file
    fpath_apr_save = io_int.save_apr_file_name()  # get file path from gui
    pyapr.io.write(fpath_apr_save, apr, parts)

    # Size of original and compressed APR files in MB
    original_file_size = os.path.getsize(fpath_apr) * 1e-6
    compressed_file_size = os.path.getsize(fpath_apr_save) * 1e-6

    # Uncompressed pixel image size (assuming 16-bit datatype)
    original_image_size = 2e-6 * apr.x_num(apr.level_max()) * apr.y_num(apr.level_max()) * apr.z_num(apr.level_max())

    print("Original File Size: {:7.2f} MB".format(original_file_size))
    print("Lossy Compressed File Size: {:7.2f} MB".format(compressed_file_size))

    # compare uncompressed pixel image size to compressed APR file sizes
    print("Original Memory Compression Ratio: {:7.2f} ".format(original_image_size/original_file_size))
    print("Lossy Memory Compression Ratio: {:7.2f} ".format(original_image_size/compressed_file_size))


if __name__ == '__main__':
    main()
