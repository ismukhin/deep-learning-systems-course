cmake_minimum_required(VERSION 3.12)
set(PROJECT_NAME "DNNStitchingPanorama")
project(${PROJECT_NAME})

option(BUILD_ASAN "Enable AddressSanitizer" OFF)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.hpp")

add_executable(stitching_sample stitching_app.cpp ${HEADERS} ${SOURCES})

include_directories(include)

find_package(onnxruntime REQUIRED)
message(STATUS "ONNXRuntime found: ${onnxruntime_VERSION}")

find_package(gflags REQUIRED)
message(STATUS "GFlags found: ${gflags_VERSION}")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

target_link_libraries(stitching_sample
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    gflags
)

if(BUILD_ASAN)
    message(STATUS "AddressSanitizer enabled")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

target_include_directories(stitching_sample PRIVATE ${OpenCV_INCLUDE_DIRS})
