# CS380 - GPGPU Programming - OpenGL exercise

## Introduction

This is the framework for the course project of 'CS380 - GPGPU Programming' at KAUST. 

## Compiling and Running the Framework (Ubuntu 18.04 Tested)
The framework is tested on Ubuntu 18.04 machines with Nvidia graphic cards (CUDA 10.0).
Install the newest version of CUDA and update your graphics driver.
A solution file is provided for Visual Studio 2017.
If you want to use a different operating system or IDE, you will have to port parts of the code yourself.
The framework uses the 'GLFW' library for setting up a window and 'glad' to initialize an OpenGL context.


1. Install required dependencies (GLFW, GLM, OpenGL)  
   `sudo apt-get install libglfw3-dev libglfw3 libglm-dev`  
   Install OpenGL is needed: `sudo apt-get install mesa-utils`

2. Compile, Link

   ```shell
   mkdir build & cd build  # or use: `cmake-gui`
   cmake ..
   make -j32
   ```

3. Run:
`./2_assignment`
the file is under `$ProjectFolder\build\2_assignment`

## Contact
Solution provider: [Guocheng Qian](https://www.gcqian.com)

