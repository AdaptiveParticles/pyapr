# PyLibAPR

[![build and deploy](https://github.com/AdaptiveParticles/PyLibAPR/actions/workflows/main.yml/badge.svg)](https://github.com/AdaptiveParticles/PyLibAPR/actions)
[![License](https://img.shields.io/pypi/l/pyapr.svg?color=green)](https://raw.githubusercontent.com/AdaptiveParticles/PyLibAPR/master/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/pyapr.svg?color=blue)]((https://python.org))
[![PyPI](https://img.shields.io/pypi/v/pyapr.svg?color=green)](https://pypi.org/project/pyapr/)
![PowerShell Gallery](https://img.shields.io/powershellgallery/p/DNS.1.1.1.1)

Python wrappers for [LibAPR]: A library for producing and processing on the Adaptive Particle Representation (APR).

The APR is an adaptive image representation designed primarily for large volumetric
microscopy datasets. By replacing pixels with particles positioned according to the
image content, it enables orders-of-magnitude compression of sparse image data
while maintaining image quality. However, unlike most compression formats, the APR
can be used directly in a wide range of processing tasks - even on the GPU!

For more detailed information about the APR and its use, see:
- [Adaptive particle representation of fluorescence microscopy images](https://www.nature.com/articles/s41467-018-07390-9) (nature communications)
- [Parallel Discrete Convolutions on Adaptive Particle Representations of Images](https://arxiv.org/abs/2112.03592) (arXiv preprint)

## Installation
For Windows 10, OSX, and Linux and Python versions 3.7-3.9 direct installation with OpenMP support should work via [pip]:
```
pip install pyapr
```
Note: Due to the use of OpenMP, it is encouraged to install as part of a virtualenv.

See [INSTALL] for manual build instructions.

## Exclusive features

In addition to providing wrappers for most of the functionality of LibAPR, we provide a number of
new features that simplify the generation and handling of the APR. For example:

* Interactive APR conversion (see [get_apr_interactive_demo](demo/get_apr_interactive_demo.py) and 
  [get_apr_by_block_interactive_demo](demo/get_apr_by_block_interactive_demo.py))
* Interactive APR z-slice viewer (see [viewer_demo](demo/viewer_demo.py))
* Interactive APR raycast (maximum intensity projection) viewer (see [raycast_demo](demo/raycast_demo.py))
* Interactive lossy compression of particle intensities (see [compress_particles_demo](demo/compress_particles_demo.py))

For further examples see the [demo scripts].

Also be sure to check out our (experimental) [napari] plugin: [napari-apr-viewer].


## License

**pyapr** is distributed under the terms of the [Apache Software License 2.0].


## Contact us

If anything is not working as you think it should, or would like it to, please get in touch with us!! Further, dont 
hesitate to contact us if you have a project or algorithm you would like to try using the APR for. We would be happy to 
assist you!


[LibAPR]: https://github.com/AdaptiveParticles/LibAPR
[pip]: https://pypi.org/project/pip/
[INSTALL]: INSTALL.md
[demo scripts]: demo
[napari]: https://napari.org
[napari-apr-viewer]: https://github.com/AdaptiveParticles/napari-apr-viewer
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
