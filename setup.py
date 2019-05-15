import sys, os
from setuptools import setup, find_packages
import subprocess
from cmake_setuptools import *


class git_clone_external(CMakeBuildExt):
    def run(self):

        tmp = False

        if not os.path.isdir( os.path.join(os.getcwd(), 'pybind11') ):
            subprocess.check_call(['git', 'clone', 'https://github.com/pybind/pybind11.git'])
        else:
            tmp = True

        if not os.path.isdir( os.path.join(os.getcwd(), 'LibAPR') ):
            subprocess.check_call(['git', 'clone', '--recursive', 'https://github.com/AdaptiveParticles/LibAPR.git'])
        else:
            tmp = True

        if tmp:
            subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])

        #not sure if this works as intended
        pyv = '{}.{}'.format(sys.version_info[0], sys.version_info[1])
        os.environ['CMAKE_COMMON_VARIABLES'] = '-DPYBIND11_PYTHON_VERSION={}'.format(pyv)

        CMakeBuildExt.run(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PyLibAPR',
    version='0.1.1',
    ext_modules=[CMakeExtension('_pyaprwrapper')],
    cmdclass={
        'build_ext': git_clone_external,
    },
    setup_requires=['cmake-setuptools'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision'
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
