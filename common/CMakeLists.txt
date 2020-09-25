set(target common)

set(common_SOURCES
        teapotpatch.cpp
        teapot.cpp
        torus.cpp
        objmesh.cpp
        cube.cpp
        plane.cpp
        teapotdata.h
        glad/glad.c
        trianglemesh.cpp trianglemesh.h aabb.h
        texture.h texture.cpp
        utils.h grid.cpp grid.h random.h
        skybox.cpp skybox.h
        stbimpl.cpp)

add_library(${target} STATIC ${common_SOURCES})

target_include_directories(${target} PUBLIC glad)
target_include_directories(${target} PUBLIC stb)
target_link_libraries(${target} PUBLIC glm)

if( UNIX AND NOT APPLE )
    target_link_libraries(${target} PUBLIC ${CMAKE_DL_LIBS})
endif()

