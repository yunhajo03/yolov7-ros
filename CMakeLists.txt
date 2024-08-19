cmake_minimum_required(VERSION 3.5)
project(yolov7_ros)


# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

install(PROGRAMS
  src/detect_ros.py  # Path to the Python script
  DESTINATION lib/${PROJECT_NAME}
)


################################################
## Declare ROS messages, services, and actions ##
################################################


# Uncomment and modify these lines if your package uses custom messages/services
# add_message_files(
#   FILES
# )


# Uncomment this if your package uses custom services
# add_service_files(
#   FILES
#   Service1.srv
# )


# Uncomment this if your package uses custom actions
# add_action_files(
#   FILES
#   Action1.action
# )



#############
## Install ##
#############


install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
