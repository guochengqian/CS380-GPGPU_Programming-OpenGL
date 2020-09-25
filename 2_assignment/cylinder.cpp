#include "cylinder.h"

#include <cstdio>
#include <cmath>

#include <glm/gtc/constants.hpp>

// nSlices : horizontal.
// nStacks : vertical
Cylinder::Cylinder(GLfloat radius, GLfloat height, GLuint nSlices, GLuint nStacks)
{
    int nVerts = (nSlices+1) * (nStacks+1);
    int elements = (nSlices+1) * (nStacks+1);

    // Verts
    std::vector<GLfloat> p(3 * nVerts);
    // Normals
    std::vector<GLfloat> n(3 * nVerts);
    // Tex coords
    std::vector<GLfloat> tex(2 * nVerts);
    // Elements
    std::vector<GLuint> el(elements);

    GLfloat nx, ny, z, s, t;
    GLfloat thetaFac = glm::two_pi<float>() / nSlices;
    GLfloat theta;

    GLuint idx = 0, tIdx = 0, eIdx=0;
    // for each xy plane
    for(GLuint i = 0; i <= nStacks; ++i)
    {
        z = -(height * 0.5f) + (float)i / nStacks * height;      // vertex position z
        t = 1.0f - (float)i / nStacks;   // top-to-bottom
        GLuint sliceStart = i * (nSlices+1);
        for(GLuint j = 0, k = 0; j <= nSlices; ++j, k += 3)
        {
            theta = j * thetaFac;
            nx = cos(theta); ny=sin(theta);
            p[idx] = radius * nx; p[idx+1] = radius * ny;p[idx+2] = z;
            n[idx] = nx; n[idx+1] = ny; n[idx+2] = 0;
            idx += 3;

            s = (GLfloat)j / nSlices;
            tex[tIdx] = s;
            tex[tIdx+1] = t;
            tIdx += 2;

            el[eIdx] = sliceStart + j;
            eIdx += 1;
        }
    }

    initBuffers(&el, &p, &n, &tex);
}
