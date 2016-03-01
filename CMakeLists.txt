cmake_minimum_required(VERSION 2.8)
project( imData )
find_package( OpenCV REQUIRED )
add_executable( imData imData.cpp )
target_link_libraries( imData ${OpenCV_LIBS} )

