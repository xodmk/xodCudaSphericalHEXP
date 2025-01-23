################################################################################
# xodmk exp - GPU Acceleration Dev
# eschei 2024_5
################################################################################

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER /usr/bin/g++)

project(xodCudaSphericalH LANGUAGES CUDA CXX)


# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)

# Set the CUDA architecture for the NVIDIA GeForce RTX 2070
# CUDA compute capability RTX2070 = 7.5	=> set(CUDA_ARCHITECTURES 75)
# CUDA compute capability RTX3090 = 8.0 => set(CUDA_ARCHITECTURES 80)
#set(CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_ARCHITECTURES 75)


# Find CUDA package
find_package(CUDA REQUIRED)
#find_package(CUDA 12.4 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(PNG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)


set(INCLUDE_DIRS 
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set(OPENGL_LIBRARIES
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${PNG_LIBRARIES}
)

include_directories(${INCLUDE_DIRS})


# Source files
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xod_opengl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xodCudaUtil.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xodCudaSphericalH.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xodCudaSpectra_mdl.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Set CUDA specific properties for the target
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        include
)


# Link against the CUDA runtime library and the chrono library
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OPENGL_LIBRARIES}
    ${CUDAToolkit_LIBRARIES}
    cufft
    PkgConfig::FFTW
)

# Add CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
