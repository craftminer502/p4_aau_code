cmake_minimum_required(VERSION 2.8.3)
project(bin-picking)

find_package(catkin REQUIRED roscpp rospy)

find_package(Boost 1.40.0 COMPONENTS filesystem system iostreams REQUIRED)
find_package(OpenCV 3.0 REQUIRED)
find_package(realsense2 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-deprecated-declarations -Wno-unused-parameter -Wno-comment")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES program
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(cv-node
${catkin_INCLUDE_DIRS}
)

add_executable(cv-node main.cpp src/cv.hpp src/features.hpp)
set_property(TARGET cv-node PROPERTY CXX_STANDARD 14)

target_link_libraries(cv-node
 ${DEPENDENCIES}
 ${catkin_LIBRARIES}
 ${OpenCV_LIBS}
 ${Boost_LIBRARIES}
 ${realsense2_LIBRARY}
)
