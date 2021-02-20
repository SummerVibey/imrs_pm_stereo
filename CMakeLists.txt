project (PatchMatchStereo)

cmake_minimum_required(VERSION 2.8)

# C++ standard
if(${CMAKE_VERSION} VERSION_LESS "3.2.0")
  set( CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3" )
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
# set(CMAKE_CXX_FLAGS "-std=c++11")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Eigen
include_directories( "/usr/local/include/eigen3" )

# OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} ) 

# Boost
find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})


set(CUDA_MIN_VERSION "7.0")
find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)
add_definitions("-DCUDA_ENABLED")
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")

# Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
# Do not show warnings if the architectures are deprecated.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")

message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                " archs: ${CUDA_ARCH_FLAGS_readable})")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -ffast-math -march=native") # extend release-profile with fast-math
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)


add_library(${PROJECT_NAME} SHARED
    PatchMatchStereo/PatchMatchStereo.cpp
    PatchMatchStereo/pms_propagation.cpp
    PatchMatchStereo/pms_util.cpp
    PatchMatchStereo/display_utils.cpp
    PatchMatchStereo/data_io.cpp
)

cuda_add_library(PatchMatchStereoCuda STATIC
    PatchMatchStereo/patch_match_cuda.cu
    PatchMatchStereo/multi_view_stereo.cu
    PatchMatchStereo/data_io.cpp
    PatchMatchStereo/display_utils.cpp
    PatchMatchStereo/texture_utils.cpp
    PatchMatchStereo/mvs_matcher_wrapper.cpp)

add_executable(patch_match_demo PatchMatchStereo/main.cpp)
target_link_libraries(patch_match_demo 
    PatchMatchStereo
    ${OpenCV_LIBS})

cuda_add_executable(stereo_cuda_test 
  PatchMatchStereo/stereo_cuda.cpp)
target_link_libraries(stereo_cuda_test
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})

cuda_add_executable(stereo_texture_test 
  PatchMatchStereo/stereo_texture.cpp)
target_link_libraries(stereo_texture_test
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})

cuda_add_executable(mvs_texture_test 
  PatchMatchStereo/mvs_texture.cpp)
target_link_libraries(mvs_texture_test
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})


cuda_add_executable(2vs_texture_test 
    PatchMatchStereo/2vs_texture.cpp)
target_link_libraries(2vs_texture_test
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})

cuda_add_executable(mvs_test 
    PatchMatchStereo/mvs_test.cpp)
target_link_libraries(mvs_test
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})

cuda_add_executable(qsinghua_demo 
    PatchMatchStereo/qsinghua_demo.cpp)
target_link_libraries(qsinghua_demo
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})

cuda_add_executable(kitti_demo 
    PatchMatchStereo/kitti_demo.cpp)
target_link_libraries(kitti_demo
    PatchMatchStereoCuda
    ${THIRD_PARTY_LIBS})

# cuda_add_executable(dtu_demo 
#     PatchMatchStereo/dtu_demo.cpp)
# target_link_libraries(dtu_demo
#     PatchMatchStereoCuda
#     ${THIRD_PARTY_LIBS})
