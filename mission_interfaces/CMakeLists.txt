cmake_minimum_required(VERSION 3.10)
project(mission_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteAdaptiveBoxGrasp.action"
  "action/ExecuteBoxGrasp.action"
  "action/ExecuteBoxPlace.action"
  "action/ExecuteDragBoxGrasp.action"
  "action/ExecuteWorkflow.action"
  "action/MoveBaseDistance.action"
  "action/PlaceBoxTest.action"
  "srv/AcquireMissionLease.srv"
  "srv/ReleaseMissionLease.srv"
  DEPENDENCIES geometry_msgs
)

ament_package()
