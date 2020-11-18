// CS 380 - GPGPU Programming, KAUST
//
// Programming Assignment #2
// Edited by Guocheng Qian

// includes c++ library
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "model/vbocube.h"
#include "model/vbomesh.h"
#include "model/vbosphere.h"
#include "resource_manager.h"
#include "camera.h"
#include "shadermodel.h"

#include "texture.h"

// args
bool multiInstance = false;
bool rotateLight = true;
bool multiPass = true;
float angle = 0.0f;
float alpha = 2.5f;

// for shader
int g_iShaderOption=1;
int g_iImgOption=0;   // image processing operation. (0 is None).
GLuint g_uFBOHandle = 0;
GLuint g_uImgVOAHandle;   // voa handle for the image processing

// window size
const unsigned int gWindowWidth = 1000;
const unsigned int gWindowHeight = 1000;
int width=gWindowWidth, height=gWindowHeight;

// mesh option constants
static const unsigned int MESH_CUBE = 0;
static const unsigned int MESH_SPHERE = 1;
static const unsigned int MESH_OBJ = 2;
// switch between meshes
unsigned int  g_uMeshOption = MESH_CUBE;

VBOCube *m_pCube; // a simple cube
VBOMesh *m_pMesh; // a more complex mesh
VBOSphere *m_pSphere; // a sphere cube

Shader obj_shader; //Shader lightsource_shader;
Camera camera; // camera
ShaderModel shadermodel; // for controlong the view matrix for object (rotate, translate)

// texture
GLuint g_iRenderOption;

// for rendering
glm::vec4 worldLight = glm::vec4(10.0f,10.0f,10.0f,1.0f); // lighting

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;
float lastX = gWindowWidth / 2.0f;
float lastY = gWindowHeight / 2.0f;
bool firstMouse = true;

void printUsage(){
    fprintf(stdout, "================================ usage ================================\n");
    fprintf(stdout, "Q/E button: switch model\n");
    fprintf(stdout, "U/O/I/K/J/L button to rotate and translate models\n");
    fprintf(stdout, "AWSD button to translate camera\n");
    fprintf(stdout, "R button: turn on/off rotation light\n");
    fprintf(stdout, "1,2,3,4,5: switch shading mode\n");
    fprintf(stdout, "1: Phong+Gouraud.\n" );
    fprintf(stdout, "2: Phong+Phong.\n" );
    fprintf(stdout, "3: Phong+Phong, Stripes.\n" );
    fprintf(stdout, "4: Phong+Phong, Lattice.\n" );
    fprintf(stdout, "5: Texture mapping, brick.\n" );
    fprintf(stdout, "6: Texture mapping, cement.\n" );
    fprintf(stdout, "7: Texture mapping, moss.\n" );
    fprintf(stdout, "Y button: turn on/off multi-pass and image processing\n");
    fprintf(stdout, "F10: Multi-pass image processing, None.\n" );
    fprintf(stdout, "F1: Multi-pass image processing, brightness.\n" );
    fprintf(stdout, "F2: Multi-pass image processing, contrast.\n" );
    fprintf(stdout, "F3: Multi-pass image processing, saturation.\n" );
    fprintf(stdout, "F4: Multi-pass image processing, blur.\n" );
    fprintf(stdout, "F5: Multi-pass image processing, sharpening.\n" );
    fprintf(stdout, "F6: Multi-pass image processing, edge.\n" );
    fprintf(stdout, "Comma: decrease alpha for image processing.\n" );
    fprintf(stdout, "Period: increase alpha for image processing.\n" );
    fprintf(stdout, "H: Help.\n" );
    fprintf(stdout, "P: Restart.\n" );
    fprintf(stdout, "=======================================================================\n");
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

void glfwErrorCallback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void APIENTRY glDebugOutput(GLenum source,
                            GLenum type,
                            GLuint id,
                            GLenum severity,
                            GLsizei length,
                            const GLchar *message,
                            const void *userParam) // OpenGL error debugging callback
{
// ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " << message << std::endl;

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;

    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
}

// query
// query GPU functionality we need for OpenGL, return false when not available
bool queryGPUCapabilitiesOpenGL()
{
    // =============================================================================
    // query and print (to console) OpenGL version and extensions:
    // - query and print GL vendor, renderer, and version using glGetString()
    printf("Q1: query and print (to console) OpenGL version and extensions:\n");
    printf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
    printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    // printf("OpenGL extensions: %s\n", glGetString(GL_EXTENSIONS));
    printf("\n=============================================================\n");

    // below is my personal reading : https://learnopengl.com/Getting-started/Hello-Triangle
    // A vertex shader is a graphics processing function used to add special effects to objects in a 3D environment by performing mathematical operations on the objects' vertex data.
    // a vertex : emerges with a different color, different textures, or a different position in space.
    // vertex shader: 3d->another 3d, process the baisc characteristics of vertex.
    // fragment shader: mainly for the color of the pixel/fragment

    // query and print GPU OpenGL limits (using glGet(), glGetInteger() etc.):
    // - maximum number of vertex shader attributes
    // - maximum number of varying floats
    // - number of texture image units (in vertex shader and in fragment shader, respectively)
    // - maximum 2D texture size
    // - maximum 3D texture size
    // - maximum number of draw buffers //define an array of buffers into which outputs from the fragment shader data will be written. I
    // =============================================================================
    printf("Q2: query and print GPU OpenGL limits (using glGet(), glGetInteger() etc.):\n");
    GLint result;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &result);
    printf("maximum number of vertex shader attributes: %d\n", result);

    glGetIntegerv(GL_MAX_VARYING_FLOATS, &result);
    printf("maximum number of varying floats: %d\n", result);

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &result);
    printf("number of texture image units in fragment shader: %d\n", result);

    glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &result);
    printf("number of texture image units in vertex shader: %d\n", result);

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &result);
    printf("number of 2D texture size: %d\n", result);

    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &result);
    printf("number of 3D texture size: %d\n", result);

    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &result);
    printf("number of draw buffers: %d\n", result);
    printf("\n=============================================================\n");
    return true;
}

// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
    printf("Q3: query GPU functionality we need for CUDA, return false when not available:\n");

    // Device Count
    int devCount;

    // Get the Device Count
    cudaGetDeviceCount(&devCount);

    // Print Device Count
    printf("Device(s): %i\n", devCount);

    // query and print CUDA functionality:
    // - CUDA device properties for every found GPU using cudaGetDeviceProperties():
    //   - device name
    //   - compute capability (The Compute Capability describes the features supported by a CUDA hardware.)
    //   - multi-processor count
    //   - clock rate (clock frequency in kilohertz; Also known as engine clock, GPU clock speed indicates how fast the cores of a graphics processing unit (GPU) are. The function of these cores is to render graphics; therefore, the higher the GPU clock speed, the faster the processing.)
    //   - total global memory
    //   - shared memory per block (A thread block is a programming abstraction that represents a group of threads that can be executed serially or in parallel. On the hardware side, a thread block is composed of 'warps'. )
    //   - num registers per block
    //   - warp size (in threads)  A warp is a set of 32 threads within a thread block such that all the threads in a warp execute the same instruction. Warp size is the number of threads in a warp, which is a sub-division used in the hardware implementation to coalesce memory access and instruction dispatch
    //   - max threads per block
    // =============================================================================
    for (int i = 0; i < devCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("device NO.: %d\n", i);
        printf("device name: %s\n", prop.name);
        printf("compute capability: major: %d, minor: %d\n",prop.major, prop.minor);
        printf("multi-processor count: %d\n",prop.multiProcessorCount);
        printf("clock rate (KHz): %d\n", prop.clockRate);
        printf("total global memory (Gb): %.2f\n", float(prop.totalGlobalMem / std::pow(1024, 3)));
        printf("shared memory per block: %lu\n", prop.sharedMemPerBlock);
        printf("num registers per block: %d\n", prop.regsPerBlock);
        printf("warp size (in threads): %d\n", prop.warpSize);
        printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
    printf("\n=============================================================\n");
    return true;
}



// init application
// - load application specific data
// - set application specific parameters
// - initialize stuff
bool initApplication(int argc, char **argv)
{
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    std::string version((const char *)glGetString(GL_VERSION));
    std::stringstream stream(version);
    unsigned major, minor;
    char dot;

    stream >> major >> dot >> minor;

    assert(dot == '.');
    if (major > 3 || (major == 2 && minor >= 0)) {
        std::cout << "OpenGL Version " << major << "." << minor << std::endl;
    } else {
        std::cout << "The minimum required OpenGL version is not supported on this machine. Supported is only " << major << "." << minor << std::endl;
        return false;
    }

    // default initialization
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);;
    glEnable(GL_DEPTH_TEST);    // Enable !!!

    // viewport
    glViewport(0, 0, gWindowWidth, gWindowHeight);
    return true;
}

void setupFBO() {
    
    // Generate and bind the framebuffer
    glGenFramebuffers(1, &g_uFBOHandle);
    glBindFramebuffer(GL_FRAMEBUFFER, g_uFBOHandle);


    // Bind the texture to the FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_iRenderOption, 0);

    // Create the depth buffer
    GLuint depthBuf;
    glGenRenderbuffers(1, &depthBuf);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuf);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);

    // Bind the depth buffer to the FBO
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, depthBuf);

    // Set the targets for the fragment output variables
    GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers);

    // Unbind the framebuffer, and revert to default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// one time initialization to setup the 3D scene
void setupScene()
{
    rotateLight = true;
    multiPass = false;
    g_iShaderOption=1;
    g_iImgOption=0;   // image processing operation

    camera = Camera(vec3(0.0f,0.0f,5.0f), vec3(0.0f,1.0f,0.0f));
    ResourceManager::LoadShader("shaders/vertexshader.vert", "shaders/fragmentshader.frag", nullptr, "shading");
    obj_shader= ResourceManager::GetShader("shading");
    obj_shader.Use();

    // init scene
    obj_shader.setUniform( "Material.Kd", 0.9f, 0.5f, 0.3f);
    obj_shader.setUniform( "Light.Ld", 1.0f, 1.0f, 1.0f);
    obj_shader.setUniform( "Light.Position", worldLight );
    obj_shader.setUniform( "Material.Ka", 0.9f, 0.5f, 0.3f);
    obj_shader.setUniform( "Light.La", 0.4f, 0.4f, 0.4f);
    obj_shader.setUniform( "Material.Ks", 0.8f, 0.8f, 0.8f);
    obj_shader.setUniform( "Light.Ls", 1.0f, 1.0f, 1.0f);
    obj_shader.setUniform( "Material.Shininess", 100.0f);

    obj_shader.setUniform( "ShaderType", g_iShaderOption );
    obj_shader.setUniform( "VarLightIntensity", vec3( 0.5f, 0.5f, 0.5f ));
    obj_shader.setUniform( "StripeColor", vec3( 0.5f, 0.5f, 0.0f ) );
    obj_shader.setUniform( "BackColor", vec3( 0.0f, 0.5f, 0.5f ) );
    obj_shader.setUniform( "Width", 0.5f );
    obj_shader.setUniform( "Fuzz", 0.1f );
    obj_shader.setUniform( "Scale", 10.0f );
    obj_shader.setUniform( "Threshold", vec2( 0.13f, 0.13f ) );

    // for pass
    obj_shader.setUniform( "Pass", 1 );

    // for image processing
    obj_shader.setUniform("ImgOpType", 0);
    obj_shader.setUniform("EdgeThreshold", 0.05f);
    obj_shader.setUniform( "ShaderType", 1 );
    obj_shader.setUniform( "alpha", 2.5f);

    // set up the rotation, translation for objects.
    shadermodel = ShaderModel(vec3(0.0f, 0.0f, 0.0f), vec3(30.0f, 30.0f, 0.0f));

    // init objects in the scene
    m_pCube = new VBOCube();

    // Once you are done setting up the basic rendering, you can add more complex meshes to your scene.
    m_pSphere = new VBOSphere(1.0f, 30, 30);

    // Now cubes and spheres are all nice - but if you want to render anything more complex you will need some kind of CAD model (i.e., essentially a triangle mesh stored in a file).
    m_pMesh = new VBOMesh("../../2_assignment/data/bs_ears.obj", false, true, true);

    // load textures
    // TODO: load files and create textures
    g_iRenderOption = Texture::loadTexture("../../data/textures/brick1.bmp");
    g_iRenderOption = Texture::loadTexture("../../data/textures/cement.bmp");
    g_iRenderOption = Texture::loadTexture("../../data/textures/moss.bmp");

    // Load moss texture file into channel 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_iRenderOption);

    // load to buffer. For the multi-pass.
    setupFBO(); // frame buffer
    unsigned int handle[2];
    glGenBuffers(2, handle);
    // Array for full-screen quad
    GLfloat verts[] = {
            -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f
    };
    GLfloat tc[] = {
            0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f
    };
    glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
    glBufferData(GL_ARRAY_BUFFER, 6 * 3 * sizeof(float), verts, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
    glBufferData(GL_ARRAY_BUFFER, 6 * 2 * sizeof(float), tc, GL_STATIC_DRAW);

    // Set up the vertex array object
    glGenVertexArrays( 1, &g_uImgVOAHandle);
    glBindVertexArray(g_uImgVOAHandle);

    glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
    glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray(0);  // Vertex position

    glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
    glVertexAttribPointer( (GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray(2);  // Texture coordinates

    glBindVertexArray(0);

}

/* read some background about the framework:
The renderFrame function is called every time we want to render our 3D scene.
Typically we want to render a new frame as a reaction to user input (e.g., mouse dragging the camera), or because we have some animation running in our scene.
We typically aim for 10-120 frames per second (fps) depending on the application (10fps is considered interactive for high demand visualization frameworks, 20fps is usually perceived as fluid, 30fps is for computationally highly demanding gaming, 60fps is the target for gaming, ~120fps is the target for VR).
From these fps-requirements it follows that your renderFrame method is very performance critical.
It will be called multiple times per second and needs to do all computations necessary to render the scene.
-> Only compute what you really have to compute in this function (but also not less).

Rendering one frame typically includes:
- updating buffers and variables in reaction to the time that has passed. There are typically three reasons to update something in your scene: 1. animation, 2. physics simulation, 3. user interaction (e.g., camera, render mode, application logic).
- clearing the frame buffer (we typically erase everything we drew the last frame)
- rendering each object in the scene with (a specific) shader program
*/

void shading()
{
    obj_shader.setUniform("Pass", 1);
    obj_shader.setUniform("ImgOpType", g_iImgOption);
    obj_shader.setUniform("alpha", alpha);

    if (multiPass) glBindFramebuffer(GL_FRAMEBUFFER, g_uFBOHandle);
    glEnable(GL_DEPTH_TEST);

    // clear frame buffer
    // If you have depth testing enabled you should also clear the depth buffer before each frame using GL_DEPTH_BUFFER_BIT; otherwise you’re stuck with the depth values from last frame:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // for camera move
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    obj_shader.setUniform( "ShaderType", g_iShaderOption );
    // for texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_iRenderOption);

    // pass1 shading
    if (rotateLight){
        obj_shader.Use();
        angle += 1.0f * deltaTime;
        if (angle > glm::two_pi<float>()) angle -= glm::two_pi<float>();
        vec4 lightPos = vec4(worldLight[0] * cos(angle), worldLight[1], worldLight[2] * sin(angle), 1.0f);
        obj_shader.setUniform("Light.Position", lightPos );
    }

    // render code goes here
    if (multiInstance){ // if activate multiple instance example
        for (int i=0; i<=3; i++){
            obj_shader.Use();
            // view/projection transformations
            glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)width / (float)height, 0.3f, 100.0f);
            glm::mat4 view = camera.GetViewMatrix();

            // model translation and rotation
            glm::mat4 model = glm::mat4(1.0f); // initlized by an indentity matrix.
            model = glm::translate(model, vec3(1.5*i,1.0,1.0));
            model = glm::rotate(model, glm::radians(30.0f*i), vec3(1.0f,0.0f,0.0f));
            model = glm::rotate(model, glm::radians(10.0f*i), vec3(0.0f,1.0f,0.0f));
            model = glm::rotate(model, glm::radians(10.0f*i), vec3(0.0f,0.0f,1.0f));
            glm::mat4 mv = view * model;
            obj_shader.setUniform("ModelViewMatrix", mv);
            obj_shader.setUniform("NormalMatrix", glm::mat3( vec3(mv[0]), vec3(mv[1]), vec3(mv[2]) ));
            obj_shader.setUniform("MVP", projection * mv);

            switch (g_uMeshOption)
            {
                case MESH_CUBE: m_pCube->render(); break;
                case MESH_SPHERE: m_pSphere->render(); break;
                case MESH_OBJ: m_pMesh->render(); break;
            }
        }
    }
    else{ // single instance
        obj_shader.Use();
        // view/projection transformations
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)width / (float)height, 0.3f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = shadermodel.model;
        glm::mat4 mv = view * model;
        obj_shader.setUniform("ModelViewMatrix", mv);
        obj_shader.setUniform("NormalMatrix", glm::mat3( vec3(mv[0]), vec3(mv[1]), vec3(mv[2]) ));
        obj_shader.setUniform("MVP", projection * mv);

        switch (g_uMeshOption)
        {
            case MESH_CUBE: m_pCube->render(); break;
            case MESH_SPHERE: m_pSphere->render(); break;
            case MESH_OBJ: m_pMesh->render(); break;
        }
    }
}

float gauss(float x, float sigma2 )
{
    double coeff = 1.0 / (glm::two_pi<double>() * sigma2);
    double expon = -(x*x) / (2.0 * sigma2);
    return (float) (coeff*exp(expon));
}
// for image processing (pass 2):

void imageprocessing()
{
    obj_shader.setUniform("ImgOpType", g_iImgOption);
    obj_shader.setUniform("Pass", 2);

    float weights[5], sum, sigma2 = 8.0f;

    // Compute and sum the weights
    weights[0] = gauss(0,sigma2);
    sum = weights[0];
    for( int i = 1; i < 5; i++ ) {
        weights[i] = gauss(float(i), sigma2);
        sum += 2 * weights[i];
    }

    // Normalize the weights and set the uniform
    for( int i = 0; i < 5; i++ ) {
        std::stringstream uniName;
        uniName << "Weight[" << i << "]";
        float val = weights[i] / sum;
        obj_shader.setUniform(uniName.str().c_str(), val);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_iRenderOption);

    glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);
    glm::mat4 mv = view * model;
    obj_shader.setUniform("ModelViewMatrix", mv);
    obj_shader.setUniform("NormalMatrix", glm::mat3( vec3(mv[0]), vec3(mv[1]), vec3(mv[2]) ));
    obj_shader.setUniform("MVP", projection * mv);

    // Render the full-screen quad
    glBindVertexArray(g_uImgVOAHandle);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

// render a frame
void renderFrame()
{
    shading();

    if (multiPass)
    {
        glFlush();
        // pass2 image processing
        imageprocessing();

    }


}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

    // translate model
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_RIGHT, deltaTime);

    // rotate model
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_ROT_X_PLUS, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_ROT_X_MINUS, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_ROT_Z_PLUS, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_ROT_Z_MINUS, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_ROT_Y_PLUS, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
        shadermodel.ProcessKeyboard(Model_ROT_Y_MINUS, deltaTime);

    // Switch shading
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        g_iShaderOption = 1;
        printf("Switched to Phong+Gouraud.\n");
    } else if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
        g_iShaderOption = 2;
        printf("Switched to Phong+Phong.\n");
    } else if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
        g_iShaderOption = 3;
        printf("Switched to Phong+Phong, Stripes.\n");
    } else if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
        g_iShaderOption = 4;
        printf("Switched to Phong+Phong, Lattice.\n");
    } else if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
        g_iShaderOption = 5;
        g_iRenderOption = 1;
        printf("Switched to texture mapping, brick1\n");
    } else if (key == GLFW_KEY_6 && action == GLFW_PRESS) {
        g_iShaderOption = 5;
        g_iRenderOption = 2;
        printf("Switched to texture mapping, cement.\n");
    } else if (key == GLFW_KEY_7 && action == GLFW_PRESS) {
        g_iShaderOption = 5;
        g_iRenderOption = 3;
        printf("Switched to texture mapping, moss.\n");
    }else if (key == GLFW_KEY_F10 && action == GLFW_PRESS) {
        g_iImgOption=0;
        printf("Switched to image processing, None.\n");
    }else if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
        g_iImgOption=1;
        printf("Switched to image processing, brightness.\n");
    } else if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
        g_iImgOption=2;
        printf("Switched to image processing, contrast.\n");
    } else if (key == GLFW_KEY_F3 && action == GLFW_PRESS) {
        g_iImgOption=3;
        printf("Switched to image processing, saturation.\n");
    }else if (key == GLFW_KEY_F4 && action == GLFW_PRESS) {
        g_iImgOption = 4;
        printf("Switched to image processing, blur.\n");
    } else if (key == GLFW_KEY_F5 && action == GLFW_PRESS) {
        g_iImgOption = 5;
        printf("Switched to image processing, sharpening.\n");
    } else if (key == GLFW_KEY_F6 && action == GLFW_PRESS) {
        g_iImgOption = 6;
        printf("Switched to image processing, edge.\n");
    }else if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS) {
        alpha += 0.1;
        printf("Increase alpha to %f .\n", alpha);
    }else if (key == GLFW_KEY_COMMA && action == GLFW_PRESS) {
        alpha -= 0.1;
        printf("Decrease alpha to %f .\n", alpha);
    }else if (key == GLFW_KEY_H && action == GLFW_PRESS) {
        printUsage();
    }else if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        // default initialization
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);;
        glEnable(GL_DEPTH_TEST);    // Enable !!!

        // viewport
        glViewport(0, 0, gWindowWidth, gWindowHeight);
        printf("Restart.\n");

        setupScene();
    }

        // for Changing the loaded model
    else if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        g_uMeshOption++;
        if (g_uMeshOption>2) {
            g_uMeshOption = 0;
        }
    } else if (key == GLFW_KEY_E && action == GLFW_PRESS) {
        if (g_uMeshOption == 0) {
            g_uMeshOption = 2;
        } else {
            g_uMeshOption--;
        }
    }

        // for turn on/off the rotating light.
    else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        rotateLight = !rotateLight;
    }

        // for turn on/off the multi pass and image processing
    else if (key == GLFW_KEY_Y && action == GLFW_PRESS) {
        multiPass = !multiPass;
        printf("use multi pass?: %d \n ", multiPass);
    }
}

// entry point
int main(int argc, char** argv)
{
    // set glfw error callback
    glfwSetErrorCallback(glfwErrorCallback);

    // init glfw
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // I tested on OpenGL 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    // init glfw window
    GLFWwindow* window;
    window = glfwCreateWindow(gWindowWidth, gWindowHeight, "CS380 - OpenGL and GLSL Shaders", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // make context current (once is sufficient)
    glfwMakeContextCurrent(window);

    // set GLFW callback functions
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // glfwSetCursorPosCallback(window, mouse_callback); // this is for the camera direction.
    glfwSetScrollCallback(window, scroll_callback);

    // get the frame buffer size
    glfwGetFramebufferSize(window, &width, &height);

    // init the OpenGL API (we need to do this once before any calls to the OpenGL API)
    if(!gladLoadGL()) { exit(-1); }

    // query OpenGL capabilities
    if (!queryGPUCapabilitiesOpenGL())
    {
        // quit in case capabilities are insufficient
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // query CUDA capabilities
    if(!queryGPUCapabilitiesCUDA())
    {
        // quit in case capabilities are insufficient
        glfwTerminate();
        exit(EXIT_FAILURE);
    }


    // init our application
    if (!initApplication(argc, argv)) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    printUsage();
    setupScene();

    // start traversing the main loop
    // loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // render one frame
        renderFrame();

        // swap front and back buffers
        glfwSwapBuffers(window);
        // poll and process input events (keyboard, mouse, window, ...)
        glfwPollEvents();
    }

    glfwTerminate();
    return EXIT_SUCCESS;
}
