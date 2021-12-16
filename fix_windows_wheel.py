import subprocess
import os
import sys
import glob
import shutil

#
#   This script fixes wheels in this folder for windows builds, copying files from the release directory
#   then repacking the wheel and then using delvewheel to fix dependencies.
#

def main(wheel_file, dest_dir):
    wheel_dir = os.path.dirname(wheel_file)
    wheel_name = os.path.basename(wheel_file)
    os.chdir(wheel_dir)

    #unpack the wheel
    subprocess.check_call(['wheel', 'unpack', wheel_name])

    folder = glob.glob('pyapr*/')[0]    # there should be only one

    # copy files out of the Release subdirectory
    path = os.path.join(folder, 'Release', '*')
    files_2_copy = glob.glob(path)
    for fc in files_2_copy:
        print(fc)
        shutil.copy(fc,folder)

    # remove the Release folder and its contents
    shutil.rmtree(os.path.join(folder, 'Release'))

    # repack the wheel
    subprocess.check_call(['wheel', 'pack', folder])

    # remove the unpacked directory
    shutil.rmtree(folder)

    # repair wheel
    subprocess.check_call(['delvewheel', 'repair', '--ignore-in-wheel', wheel_name])

    # copy repaired wheel to destination directory
    shutil.copy(wheel_name, dest_dir)


if __name__ == '__main__':
    _, wheel_file, dest_dir = sys.argv
    main(wheel_file, dest_dir)
