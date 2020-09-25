#pragma once

#include "trianglemesh.h"
#include "cookbookogl.h"

class Cylinder : public TriangleMesh
{
public:
    Cylinder(float rad, float height, GLuint sl, GLuint st);
};
