#!/bin/sh
set -x
travis_wait 10 choco install visualstudio2019buildtools --params "--add Microsoft.Component.MSBuild --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Llvm.Clang --add Microsoft.VisualStudio.Component.Windows10SDK.19041	--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.ComponentGroup.UWP.VC.BuildTools"
mkdir build
cd build
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg.exe install blosc:x64-windows gtest:x64-windows tiff:x64-windows hdf5:x64-windows szip:x64-windows
cd ..
powershell $vcpkg_path = Resolve-Path "\vcpkg\scripts\buildsystems\vcpkg.cmake"
powershell $cmake_arg = "-DCMAKE_TOOLCHAIN_FILE=" + $vcpkg_path
powershell $env:EXTRA_CMAKE_ARGS = $cmake_arg