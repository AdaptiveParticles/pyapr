# build instructions
the readme is a bit dated, follow these instructions for now

i) make sure the LibAPR submodule is set to track develop_joel:
```
cd LibAPR
git fetch --all
git checkout develop_joel
git pull
cd ..
```
ii) (optional) set up a virtual environment, e.g.
```
python3 -m virtualenv env
source env/bin/activate
```

iii) install necessary python packages
```
pip install cmake-setuptools numpy matplotlib pyqt5 pyqtgraph scikit-image
```

iv) build via the setup.py script, e.g.
```
CMAKE_COMMON_VARIABLES="-DPYAPR_USE_CUDA=OFF -DPYAPR_USE_OPENMP=ON" python setup.py develop
```

To build on mac with openmp support the README instructions may (should) still apply
```
CPPFLAGS="-I/usr/local/opt/llvm/include" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CXX="/usr/local/opt/llvm/bin/clang++" CC="/usr/local/opt/llvm/bin/clang" CMAKE_COMMON_VARIABLES="-DPYAPR_USE_CUDA=OFF -DPYAPR_USE_OPENMP=ON" python setup.py develop 
```