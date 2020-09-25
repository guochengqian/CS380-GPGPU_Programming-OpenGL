set(target 1_assignment)

set( 1_assignment_SOURCES
        main_assignment1.cpp
        )

add_executable( ${target} ${1_assignment_SOURCES} )

target_link_libraries( ${target}
        PRIVATE
        common
        glfw
        ${OPENGL_gl_LIBRARY}
        ${CUDA_LIBRARIES}
        )

