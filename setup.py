# -*- coding: utf-8 -*-
import os
import sys
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.

        # cmake_args = []
        # cmake_args = [
            # "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            # "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            # "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            # "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        # ]
        build_args = []

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            x = 1
        else:
            # cmake_args += ["-A"]
            #  cmake_args +=  ["x64"]
            # cmake_args += ["-DCMAKE_TOOLCHAIN_FILE=\"C:/Users/bevan/APR/LibAPR/vcpkg/scripts/buildsystems/vcpkg.cmake\""]
            # cmake_args += ["-DVCPKG_TARGET_TRIPLET=x64-windows"]
            # cmake_args += ["-T ClangCL"]

            cmake_args = []
            # cmake_args += ["cmake"]
            # cmake_args += [".."]
            cmake_args += ["-A"]
            cmake_args +=  ["x64"]
            cmake_args += ["-DCMAKE_TOOLCHAIN_FILE=C:\\Users\\bevan\\APR\\LibAPR\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake"]
            cmake_args += ["-DVCPKG_TARGET_TRIPLET=x64-windows"]
            cmake_args += ["-T"]
            cmake_args += ["ClangCL"]

            x = 1

        build_args = "--config Release"

        print(cmake_args)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)


        # subprocess.check_call(
        #     ["cmake", ext.sourcedir], cwd=self.build_temp
        # )

        print("******************************************************************************")
        print( ext.sourcedir)
        print( self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".","--config","Release"], cwd=self.build_temp
        )



setup(
    name='PyLibAPR',
    version='0.1.1',
    ext_modules=[CMakeExtension('_pyaprwrapper')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    setup_requires=['cmake-setuptools'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-image',
        'libtiff',
        'PyQt5',
        'pyqtgraph',
        'matplotlib'
    ],
    description='Python wrappers for LibAPR',
    long_description="long_description",
    url='https://github.com/joeljonsson/PyLibAPR',
    author='Joel Jonsson, Bevan Cheeseman',
    author_email='jonsson@mpi-cbg.de',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='LibAPR, PyLibAPR, APR',
    zip_safe=False
)
