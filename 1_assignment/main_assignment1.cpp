// CS 380 - GPGPU Programming, KAUST
//
// Programming Assignment #1

// includes c++ library
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>


#define GLFW_INCLUDE_NONE	// this is suggested by the official guide to make sure there will be no header conflicts
#include "glad/glad.h" 
#include "GLFW/glfw3.h"
// includes, cuda
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// window size
const unsigned int gWindowWidth = 512;
const unsigned int gWindowHeight = 512;


// glfw error callback
void glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

// OpenGL error debugging callback
void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	const GLchar *message,
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




// query GPU functionality we need for OpenGL, return false when not available
bool queryGPUCapabilitiesOpenGL()
{
	// =============================================================================
	//TODO:
	// for all the following:
	// read up on concepts that you do not know and that are needed here!
	//
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
	
	// =============================================================================
	//TODO:
	// for all the following:
	// read up on concepts that you do not know and that are needed here!
	// Blocs, warps and threads

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
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, gWindowWidth, gWindowHeight);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
		
	return true;
}
 

// render a frame
void renderFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// render code goes here

}



// =============================================================================
//TODO: read background info about the framework: 
//
//In graphics applications we typically need to create a window where we can display something.
//Window-APIs are quite different on linux, mac, windows and other operating systems. 
//We use GLFW (a cross-platform library) to create a window and to handle the interaction with this window.
//It is a good idea to spend a moment to read up on GLFW:
//https://www.glfw.org/
//
//We will use it to get input events - such as keyboard or mouse events and for displaying frames that have been rendered with OpenGL.
//You should make yourself familiar with the API and read through the code below to understand how it is used to drive a graphics application.
//In general try to understand the anatomy of a typical graphics application!
// =============================================================================


//Implement mouse and keyboard callbacks!
//Print information about the events on std::cout
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	std::cout << "key_callback: key " << key << ", scancode " << scancode << ", action " << action << ", mods "<< mods << std::endl;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}
static void framebuffer_callback(GLFWwindow* window, int x, int y)
{
	std::cout << "framebuffer_callback: x " << x << ", y " << y << std::endl;
}
static void mousebutton_callback(GLFWwindow* window, int button, int action, int mods)
{
	std::cout << "mousebutton_callback: button " << button << ", action " << action << ", action " << action << ", mods " << mods << std::endl;
}
static void cursorposcall_back(GLFWwindow* window, double x, double y)
{
	std::cout << "cursorposcall_back: x " << x << ", y " << y << std::endl;
}
static void scroll_callback(GLFWwindow* window, double x, double y)
{
	std::cout << "scroll_callback: x " << x << ", y " << y << std::endl;
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

	// init glfw window 
	GLFWwindow* window;
	window = glfwCreateWindow(gWindowWidth, gWindowHeight, "CS380 - GPGPU - OpenGL Window", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// set GLFW callback functions 
	// =============================================================================
	//TODO: read up on certain GLFW callbacks which we will need in the future. 
	// Get an understanding for what a 'callback' is. Questions you should be able to answer include:
	// What is a callback? When is a callback called? How do you use a callback in your application? What are typical examples for callbacks in the context of graphics applications?
	
	// A callback is any executable code that is passed as an argument to other code; that other code is expected to call back (execute) the argument at a given time.
	// A callback is called when a speicified condition is met. 
	// A typical application in graphics application: click the mouse and then rotate the viewpoint along a direction
	
	
	//Have a look at the following examples:
	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_callback);
	glfwSetMouseButtonCallback(window, mousebutton_callback);
	glfwSetCursorPosCallback(window, cursorposcall_back);
	glfwSetScrollCallback(window, scroll_callback);
	// ...

	//Implement mouse and keyboard callbacks!
	//Print information about the events on std::cout
	// =============================================================================

	// make context current (once is sufficient)
	glfwMakeContextCurrent(window);
	
	// get the frame buffer size
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// init the OpenGL API (we need to do this once before any calls to the OpenGL API)
	gladLoadGL();  //If you are using an extension loader library to access modern OpenGL then this is when to initialize it, as the loader needs a current context to load from.

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


