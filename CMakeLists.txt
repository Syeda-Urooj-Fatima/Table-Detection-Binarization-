cmake_minimum_required(VERSION 3.5)
project(Practice)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES test.cpp check.cpp main.cpp binarization.cpp predict_boxes.cpp)
add_executable(rgb ${SOURCE_FILES})
include_directories(/usr/local/include/)
target_link_libraries(rgb opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

