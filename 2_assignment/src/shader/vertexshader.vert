#version 400

// Reference: GLSL CookBook, Chapter 2.

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;
layout (location = 2) in vec2 TextureCoord;

out vec3 LightIntensity;
struct LightInfo {
	vec4 Position; // Light position in eye coords.
	vec3 La; // Ambient light intensity
	vec3 Ld; // Diffuse light intensity
	vec3 Ls; // Specular light intensity
};

uniform LightInfo Light;
struct MaterialInfo {
	vec3 Ka; // Ambient reflectivity
	vec3 Kd; // Diffuse reflectivity
	vec3 Ks; // Specular reflectivity
	float Shininess; // Specular shininess factor
};

uniform MaterialInfo Material;

uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;

uniform int ShaderType;



out vec3 Position;
out vec3 Normal;
out vec2 TexCoord;

void main()
{
	if( ShaderType == 1 ) {
		vec3 tnorm = normalize( NormalMatrix * VertexNormal);
		vec4 eyeCoords = ModelViewMatrix *
		vec4(VertexPosition,1.0);
		vec3 s = normalize(vec3(Light.Position - eyeCoords));
		vec3 v = normalize(-eyeCoords.xyz);
		vec3 r = reflect( -s, tnorm );
		vec3 ambient = Light.La * Material.Ka;
		float sDotN = max( dot(s,tnorm), 0.0 );
		vec3 diffuse = Light.Ld * Material.Kd * sDotN;
		vec3 spec = vec3(0.0);
		if( sDotN > 0.0 )
		spec = Light.Ls * Material.Ks *
		pow( max( dot(r,v), 0.0 ), Material.Shininess );

		LightIntensity = ambient + diffuse + spec;
		gl_Position = MVP * vec4(VertexPosition,1.0);
		
	} else if( ShaderType == 2 || ShaderType == 3 || ShaderType == 4 || ShaderType == 5 ) {
		Normal = normalize( NormalMatrix * VertexNormal);
		Position = vec3( ModelViewMatrix * vec4(VertexPosition,1.0) );

		if( ShaderType == 3 || ShaderType == 4 || ShaderType == 5 ) {
			TexCoord =  TextureCoord;
		}

		gl_Position = MVP * vec4(VertexPosition,1.0);
	}
}