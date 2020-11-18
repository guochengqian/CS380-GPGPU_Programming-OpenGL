set(target common)

set(common_SOURCES
        camera.h
        shader.cpp shader.h
#        resource_manager.cpp
#        texture.cpp

        model/cube.cpp
        model/sphere.cpp model/sphere.h
        model/teapotpatch.cpp
        model/teapot.cpp
        model/torus.cpp
        model/plane.cpp
        model/teapotdata.h
        model/objmesh.cpp
        model/trianglemesh.cpp
        model/skybox.cpp
        model/vbomesh.cpp
        model/vbocube.cpp
        model/vbosphere.cpp

        aabb.h
        utils.h grid.cpp random.h
        stbimpl.cpp

        glad/glad.c

        CImg/CImg.h
        )

add_library(${target} STATIC ${common_SOURCES})

target_include_directories(${target} PUBLIC glad)
target_include_directories(${target} PUBLIC stb)
target_link_libraries(${target} PUBLIC glm)

if( UNIX AND NOT APPLE )
    target_link_libraries(${target} PUBLIC ${CMAKE_DL_LIBS})
endif()

#include_directories(  opencv-4.5.0 )
#add_subdirectory( opencv-4.5.0 )
