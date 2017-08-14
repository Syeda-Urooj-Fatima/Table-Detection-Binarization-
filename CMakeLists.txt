cmake_minimum_required(VERSION 3.5)
project(Practice)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES )
add_executable(gray binarization.cpp)
add_executable(rgb binarization_RGB.cpp)
target_link_libraries(gray opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(rgb opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
