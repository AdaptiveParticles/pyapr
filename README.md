# PyLibAPR

Externalized Python wrappers for LibAPR

## Dependencies

* HDF5 1.8.20 or higher
* OpenMP > 3.0 (optional, but suggested)
* CMake 3.6 or higher
* LibTIFF 4.0 or higher

## Building

The repository requires sub-modules, so the repository needs to be cloned recursively:

```
git clone --recursive https://github.com/joeljonsson/PyLibAPR.git
```

If you need to update your clone at any point later, run

```
git pull
git submodule update
```

### Building manually

The library can be built manually using CMake. Note that, when built in this way, the path to the build folder must be specified when importing the library in a Python script. 

#### Building on Linux

On Ubuntu, install the `cmake`, `build-essential`, `libhdf5-dev` and `libtiff5-dev` packages (on other distributions, refer to the documentation there, the package names will be similar). OpenMP support is provided by the GCC compiler installed as part of the `build-essential` package.

In the directory of the cloned repository, run

```
mkdir build
cd build
cmake ..
make
```

This will create the `pyApr.so` library in the `build` directory.

#### Building on OSX

On OSX, install the `cmake`, `hdf5` and `libtiff`  [homebrew](https://brew.sh) packages and have the [Xcode command line tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) installed.

If you want to compile with OpenMP support, also install the `llvm` package (this can also be done using homebrew), as the clang version shipped by Apple currently does not support OpenMP.

In the directory of the cloned repository, run

```
mkdir build
cd build
cmake ..
make
```

This will create the `pyApr.so` library in the `build` directory.

In case you want to use the homebrew-installed clang (OpenMP support), modify the call to `cmake` above to

```
CC="/usr/local/opt/llvm/bin/clang" CXX="/usr/local/opt/llvm/bin/clang++" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CPPFLAGS="-I/usr/local/opt/llvm/include" cmake ..
```

### Build using the setup script

It is recommended to do this in a virtual environment. First, install the dependencies for your operating system as described in the instructions for the manual build. Additionally, install `cmake-setuptools`, e.g. through

```
pip install cmake-setuptools
```

Then simply run the `setup.py` script:

```
python setup.py install
```

To use the homebrew-installed clang for OpenMP support on OSX, modify the call above to

```
CPPFLAGS="-I/usr/local/opt/llvm/include" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CXX="/usr/local/opt/llvm/bin/clang++" CC="/usr/local/opt/llvm/bin/clang" python setup.py install 
```
