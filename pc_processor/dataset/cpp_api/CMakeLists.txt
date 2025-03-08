cmake_minimum_required(VERSION 3.5)
project(occupancy_api)

set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++14)
add_definitions(-g)
add_definitions(-DDEBUG_LEVEL=0)
add_compile_options(-W)
add_compile_options(-std=c++14)


set(Python3_ROOT_DIR "/usr/local/lib/python3.8/dist-packages")
find_package(Eigen3 REQUIRED)
find_package(pybind11 CONFIG REQUIRED
        PATHS ${Python3_ROOT_DIR}
        )

#========================
# modules
#========================
pybind11_add_module(${PROJECT_NAME}
        occupancy_interface.cpp)
message(${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")

target_include_directories(${PROJECT_NAME} PUBLIC
        ${EIGEN3_INCLUDE_DIR}
        )