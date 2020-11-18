set(target 2_assignment)

set( 2_assignment_SOURCES
        ./src/main_assignment2.cpp
        ./src/texture.cpp
)


add_executable( ${target} ${2_assignment_SOURCES} )

target_link_libraries( ${target}
        PRIVATE
        common
        glfw
        ${OPENGL_gl_LIBRARY}
        ${CUDA_LIBRARIES}
        )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/shader DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


