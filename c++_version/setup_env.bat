@echo off
setlocal

echo ===================================================
echo     Geometric Gaze Setup (C++ dependencies)
echo ===================================================
echo.

if not exist vcpkg (
    echo [1/3] Cloning vcpkg - Package Manager...
    git clone https://github.com/microsoft/vcpkg.git
    echo Bootstrapping vcpkg...
    call vcpkg\bootstrap-vcpkg.bat
)

echo.
echo [2/3] Installing OpenCV[dnn] and Eigen3...
echo Note: This process requires CMake and Visual Studio C++ Build Tools.
echo Building OpenCV from source takes a significant amount of time. Please be patient.
vcpkg\vcpkg.exe install opencv[dnn]:x64-windows eigen3:x64-windows

echo.
echo [3/3] Generating CMake Project...
if not exist build mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cd ..

echo.
echo ===================================================
echo Setup Complete! 
echo Open the "build" folder in Visual Studio or run:
echo    cd build
echo    cmake --build . --config Release
echo ===================================================
pause
