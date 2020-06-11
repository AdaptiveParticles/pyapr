from setuptools import setup, find_packages
from cmake_setuptools import *

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PyLibAPR',
    version='0.1.1',
    ext_modules=[CMakeExtension('_pyaprwrapper')],
    cmdclass={
        'build_ext': CMakeBuildExt,
    },
    setup_requires=['cmake-setuptools'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyqtgraph',
        'matplotlib'
    ],
    description='Python wrappers for LibAPR',
    long_description=long_description,
    url='https://github.com/joeljonsson/PyLibAPR',
    author='Joel Jonsson',
    author_email='jonsson@mpi-cbg.de',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='LibAPR, PyLibAPR, APRNet',
    zip_safe=False
)
