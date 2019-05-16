import sys, os
from setuptools import setup, find_packages
import subprocess
from cmake_setuptools import *


def check_submodules():
    """ verify that the submodules are checked out and clean
        use `git submodule update --init`; on failure
    """
    if not os.path.exists('.git'):
        return
    with open('.gitmodules') as f:
        for l in f:
            if 'path' in l:
                p = l.split('=')[-1].strip()
                if not os.path.exists(p):
                    raise ValueError('Submodule %s missing' % p) #call git clone if this occurs???

    proc = subprocess.Popen(['git', 'submodule', 'status'],
                            stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith('-') or line.startswith('+'):
            raise ValueError('Submodule not clean: %s' % line)


class git_clone_external(CMakeBuildExt):
    def run(self):

        check_submodules()

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
