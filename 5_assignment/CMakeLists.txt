cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project (assignment LANGUAGES CUDA CXX )



# Adding cuda libraries
find_package(CUDA)


# Adding compile units to executable
CUDA_ADD_EXECUTABLE(assignment
		./src/CS380_prog.cpp
		./src/matrixkernels.cu
		)


# adding common folder
include_directories(../common/ ../common/CImg)

# Set project assignment as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT assignment)


include_directories(${CUDA_INCLUDE_DIRS})

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	target_link_libraries(assignment ${CUDA_LIBRARIES} cuda)
else()
	target_link_libraries(assignment ${CUDA_LIBRARIES} cuda X11)
endif()

# C++ Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Setting up CUDA compiler (NVCC) flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

# Adding glfw3
#find_package(glfw3 REQUIRED)
#add_subdirectory(../common/glfw-3.3 ../common/glfw-3.3/build)
#target_link_libraries(assignment glfw)

target_compile_features(assignment PUBLIC cxx_std_11)

