=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #3
Image Processing with GLSL 

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

Tasks:
1. Texture Mapping
- read different textures from the folder "data/textures", use the bmpreader or any library you like. you can also add new textures. 
- render a box with a texture assigned to it.
- load multiple textures and add a keyboard shortcut to switch between them.

2. Image Processing with GLSL using Multi-Pass Rendering
One invocation (draw call) of the OpenGL graphics pipeline is often called a pass. 
Single-pass rendering needs only one pass to compute the output image.
Multi-pass rendering requires multiple invocations of the graphics pipeline to compute one output image. 
Intermediate results are stored in frame buffer objects (FBOs) that are not displayed.

This example requires two rendering passes.
In the first pass use a frame buffer object to first apply an image processing operation on a texture. 
In the second pass use this modified texture for conventional texture mapping.

- Pass 1: render the processed image into a target texture attached to an OpenGL FBO. 
You will need to: 1) make a rectangle, 2) map the input texture to it, 3) perform image processing operations, and 4) render it to a target texture (in an FBO) that is the same size as the input texture.
Perform the image processing operation(s) in the fragment shader of this rendering pass using normalized texture coordinate arithmetics. 
You will need to pass the size (or the distance between pixels) of the input texture as uniform variable to the shader.
- Pass 2: use the resulting texture (from pass 1) and apply it as texture when rendering a 3D object.

Implement the following image processing operations:
- Per-pixel operations: implement operations to adjust a) brightness, b) contrast, and c) saturation of the texture (Chapter 19.5. [1]).
- Filter operations: Implement filters for d) smoothing, e) edge detection, and f) sharpening (Chapter 19.7. [1])

3. User Interaction
All settings must be accessible from your program without the need to modify the source code. 
Reuse your mouse/camera interaction from the last assignment.
3.a) Use key mappings to switch between the image processing operations
3.b) Use additional key mappings as you see fit for increasing and decreasing certain parameters of the image processing operations (+, -, 1, 2, 3, ...) 

4. Document the results of 1. and 2., as well as the key mappings of 3. in the report.

BONUS: Use a compute shader for the computations of Pass 1.


=====================================================================
References and Acknowledgments:
[1] OpenGL Shading Language, by Rost, Randi J; Licea-Kane, Bill, 2010:
https://learning.oreilly.com/library/view/opengl-shading-language/9780321669247/ch19.html


The provided textures are from http://www.grsites.com/