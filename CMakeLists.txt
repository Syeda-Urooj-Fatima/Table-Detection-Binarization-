cmake_minimum_required(VERSION 3.5)
project(Practice)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES binarization.cpp)
add_executable(a.out ${SOURCE_FILES})
target_link_libraries(a.out opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
