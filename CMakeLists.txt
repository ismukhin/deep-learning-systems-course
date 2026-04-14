cmake_minimum_required(VERSION 3.12)
project(MyProject)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

add_executable(my_app panorama_stitching_sample.cpp)
add_executable(my_probe superpoint.cpp)

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)


target_link_libraries(my_app
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so
)

target_link_libraries(my_probe
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)


target_compile_options(my_app PRIVATE ${ONNXRUNTIME_COMPILE_FLAGS})
#target_compile_options(my_app PUBLIC -fsanitize=address -fno-omit-frame-pointer -g)
#target_link_options(my_app PUBLIC -fsanitize=address)
target_compile_options(my_probe PRIVATE ${ONNXRUNTIME_COMPILE_FLAGS})

target_link_options(my_app PRIVATE ${ONNXRUNTIME_LINK_FLAGS})
target_link_options(my_probe PRIVATE ${ONNXRUNTIME_LINK_FLAGS})

target_include_directories(my_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(my_probe PRIVATE ${OpenCV_INCLUDE_DIRS})
