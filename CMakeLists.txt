cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

project (CS380)

if(MSVC)
    add_definitions(/DNOMINMAX)
endif()

if(UNIX)
    message(STATUS "Running Linux compiling. Options [GLVND] for modern OpenGL or [LEGACY] for historical OpenGl.")
    message(STATUS "Setting GLVND option.")
    set(OpenGL_GL_PREFERENCE GLVND)
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CS380)
# C++ Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Setting up CUDA compiler (NVCC) flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_HOST_COMPILER clang++)

# Adding cuda libraries
find_package(CUDA 10.1)
include_directories(${CUDA_INCLUDE_DIRS})
SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}  cuda X11")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)

find_package( glm CONFIG REQUIRED )
find_package( glfw3 CONFIG REQUIRED )
find_package( OpenGL REQUIRED )

include_directories( common)
include_directories( data)

add_subdirectory( common )
add_subdirectory( 1_assignment )
add_subdirectory( 2_assignment )
add_subdirectory( 3_assignment )
add_subdirectory( 4_assignment )


