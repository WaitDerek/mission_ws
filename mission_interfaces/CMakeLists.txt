cmake_minimum_required(VERSION 3.10)
project(mission_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteGrasp.action"
  "action/ExecutePlace.action"
  "action/ExecuteBinGrasp.action"
  "action/ExecuteBinPlace.action"
  DEPENDENCIES geometry_msgs
)

ament_package()
