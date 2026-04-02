cmake_minimum_required(VERSION 3.15)
project(GazeEstimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)
find_package(Eigen3 CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

add_executable(GazeEstimation src/main.cpp)

target_include_directories(GazeEstimation PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(GazeEstimation PRIVATE 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    onnxruntime::onnxruntime
)

# Copy DLLs on Windows (vcpkg does this usually, but safe to have)
if(MSVC)
    add_custom_command(TARGET GazeEstimation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:onnxruntime::onnxruntime>
        $<TARGET_FILE_DIR:GazeEstimation>
    )
endif()
