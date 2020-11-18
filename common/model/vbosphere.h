#ifndef VBOSPHERE_H
#define VBOSPHERE_H


#define PI 3.141592653589793
#define TWOPI 6.2831853071795862
#define TWOPI_F 6.2831853f
#define TO_RADIANS(x) (x * 0.017453292519943295)
#define TO_DEGREES(x) (x * 57.29577951308232)


#include "glad/glad.h" 

// include glfw library: http://www.glfw.org/
#include <GLFW/glfw3.h>


#include <cstdio>


class VBOSphere //: public Drawable
{
private:
    unsigned int vaoHandle;
    GLuint nVerts, elements;
    float radius, slices, stacks;

    void generateVerts(float * , float * ,float *, GLuint *);

public:
    VBOSphere(float, int, int);

    void render() const;

    int getVertexArrayHandle();
};

#endif // VBOSPHERE_H
