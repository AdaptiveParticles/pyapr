### Dependencies

[LibAPR](https://github.com/AdaptiveParticles/LibAPR) is included as a submodule, and built alongside the wrappers.
This requires the following packages:

* HDF5 1.8.20 or higher
* OpenMP > 3.0 (optional, but recommended)
* CMake 3.6 or higher
* LibTIFF 4.0 or higher

The Python library additionally requires Python 3, and the packages listed in [requirements.txt](requirements.txt).

### Installing dependencies on Linux

On Ubuntu, install the `cmake`, `build-essential`, `libhdf5-dev` and `libtiff5-dev` packages (on other distributions,
refer to the documentation there, the package names will be similar). OpenMP support is provided by the GCC compiler
installed as part of the `build-essential` package.

### Installing dependencies on OSX

On OSX, install the `cmake`, `hdf5` and `libtiff`  [homebrew](https://brew.sh) packages and have the
[Xcode command line tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) installed.
If you want to compile with OpenMP support, also install the `llvm` package (this can also be done using homebrew),
as the clang version shipped by Apple currently does not support OpenMP.

### Note for windows users

Please see https://github.com/AdaptiveParticles/LibAPR for the latest windows install instructions.

## Building

The repository requires submodules, so the repository needs to be cloned recursively:

```
git clone --recursive https://github.com/AdaptiveParticles/PyLibAPR.git
```

It is recommended to use a virtual environment, such as `virtualenv`. To set this up, use e.g.

```
pip3 install virtualenv
python3 -m virtualenv myenv
source myenv/bin/activate
```

The required Python packages can be installed via the command
```
pip install -r requirements.txt 
```

Once the dependencies are installed, PyLibAPR can be built via the setup.py script:
```
python setup.py install
```

### CMake build options

There are two CMake options that can be given to enable or disable OpenMP and CUDA:

| Option | Description | Default value |
|:--|:--|:--|
| PYAPR_USE_OPENMP | Enable multithreading via OpenMP | ON |
| PYAPR_USE_CUDA | Build available CUDA functionality | OFF |
| PYAPR_PREFER_EXTERNAL_LIBAPR | Use an installed version of LibAPR (if found) rather than building it from submodules | OFF |

When building via the setup.py script, these options can be set via the environment variable `EXTRA_CMAKE_ARGS`. For example,
```
EXTRA_CMAKE_ARGS="-DPYAPR_USE_OPENMP=OFF -DPYAPR_USE_CUDA=OFF" python setup.py install
```
will install the package with both OpenMP and CUDA disabled. If building from an installed version of LibAPR in a non-standard
location, help CMake find it, e.g., by passing `-DCMAKE_PREFIX_PATH=/path/to/APR`.

### OpenMP support on OSX

To use the homebrew-installed clang for OpenMP support on OSX, modify the call above to
```
CPPFLAGS="-I/usr/local/opt/llvm/include" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CXX="/usr/local/opt/llvm/bin/clang++" CC="/usr/local/opt/llvm/bin/clang" python setup.py install 
```