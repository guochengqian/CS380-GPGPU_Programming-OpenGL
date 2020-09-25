set(target 2_assignment)

set( 2_assignment_SOURCES
        main_assignment2.cpp
        resource_manager.cpp
        shader.cpp
        texture.cpp
        vbocube.cpp
        vbomesh.cpp
        cylinder.cpp
        sphere.cpp
)


add_executable( ${target} ${2_assignment_SOURCES} )

target_link_libraries( ${target}
        PRIVATE
        common
        glfw
        ${OPENGL_gl_LIBRARY}
        ${CUDA_LIBRARIES}
        )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


