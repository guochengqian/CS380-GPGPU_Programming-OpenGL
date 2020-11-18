cmake_minimum_required(VERSION 3.15)
set(target assignment_3)
project (${target})

set( assignment_3_SOURCES
		./src/main_assignment3.cpp
		./src/texture.cpp
		./src/resource_manager.cpp
		)


add_executable( ${target} ${assignment_3_SOURCES} )

target_link_libraries( ${target}
		PRIVATE
		common
		glfw
		${OPENGL_gl_LIBRARY}
		${CUDA_LIBRARIES}
		)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


