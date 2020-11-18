#ifndef ShaderModel_H
#define ShaderModel_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

const float Trans_SENSITIVITY =  2.5f;
const float ROT_SENSITIVITY =  30.0f;
const float Scale_SENSITIVITY = 0.5f;

// Defines several possible options for ShaderModel movement. Used as abstraction to stay away from window-system specific input methods
enum ShaderModel_Movement {
    Model_FORWARD,
    Model_BACKWARD,
    Model_LEFT,
    Model_RIGHT,
    Model_ROT_X_PLUS,
    Model_ROT_X_MINUS,
    Model_ROT_Z_PLUS,
    Model_ROT_Z_MINUS,
    Model_ROT_Y_PLUS,
    Model_ROT_Y_MINUS,
};

// An abstract ShaderModel class that processes input and calculates the corresponding translation, rotatation and scaling for use in OpenGL
class ShaderModel
{
public:
    // camera Attributes
    glm::vec3 Translation;
    glm::vec3 Rotation;
    float Scale;
    glm::mat4 model;

    // constructor with vectors
    ShaderModel(glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 rotation = glm::vec3(0.0f, 0.0f, 0.0f), float scale = float(1.0f))
    {
        Translation = translation;
        Rotation = rotation;
        Scale = scale;
        model = glm::mat4(1.0f);
        updateShaderModelVectors();
    }


    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(ShaderModel_Movement direction, float deltaTime)
    {
        float velocity = Trans_SENSITIVITY * deltaTime;
        if (direction == Model_FORWARD)
            Translation[2] += velocity;
        if (direction == Model_BACKWARD)
            Translation[2] -= velocity;
        if (direction == Model_LEFT)
            Translation[0] -= velocity;
        if (direction == Model_RIGHT)
            Translation[0] += velocity;

        if (direction == Model_ROT_X_PLUS)
            Rotation[0] += velocity;
        if (direction == Model_ROT_X_MINUS)
            Rotation[0] -= velocity;
        if (direction == Model_ROT_Z_PLUS)
            Rotation[2] += velocity;
        if (direction == Model_ROT_Z_MINUS)
            Rotation[2] -= velocity;
        if (direction == Model_ROT_Y_PLUS)
            Rotation[1] += velocity;
        if (direction == Model_ROT_Y_MINUS)
            Rotation[1] -= velocity;
        updateShaderModelVectors();
    }
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateShaderModelVectors()
    {
        // calculate the new Front vector
        model = glm::mat4(1.0f); // initlized by an indentity matrix.
        model = glm::translate(model, Translation);
        if (Rotation[0] != 0.0f){
            model = glm::rotate(model, glm::radians(Rotation[0]), vec3(1.0f,0.0f,0.0f));
        }
        if (Rotation[1] != 0.0f){
            model = glm::rotate(model, glm::radians(Rotation[1]), vec3(0.0f,1.0f,0.0f));
        }
        if (Rotation[2] != 0.0f){
            model = glm::rotate(model, glm::radians(Rotation[2]), vec3(0.0f,0.0f,1.0f));
        }
    }
private:

};
#endif