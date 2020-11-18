#version 400

// Reference: GLSL CookBook, Chapter 2.

in vec3 LightIntensity;
layout( location = 0 ) out vec4 FragColor;

uniform int ShaderType;
in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 VarLightIntensity;

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

uniform vec3 StripeColor;
uniform vec3 BackColor;
uniform float Width;
uniform float Fuzz;
uniform float Scale;
uniform vec2 Threshold;


vec3 ads( vec3 col )
{
	vec3 n = normalize( Normal );
	vec3 s = normalize( vec3(Light.Position) - Position );
	vec3 v = normalize(vec3(-Position));
	vec3 r = reflect( -s, n );
	return VarLightIntensity * ( col * ( Material.Ka + Material.Kd * max( dot(s, n), 0.0 ) ) + 
		Material.Ks * pow( max( dot(r,v), 0.0 ), Material.Shininess ) );
}

void main() {

	if( ShaderType == 1 ) {
		FragColor = vec4( LightIntensity, 1.0 );
	} else if( ShaderType == 2 ) {
		FragColor = vec4( ads( vec3(1.0f, 1.0f, 1.0f ) ), 1.0 );
	} else if( ShaderType == 3 ) { 

		float scaled_t = fract( TexCoord.t * Scale );
        float frac1 = clamp( scaled_t / Fuzz, 0.0, 1.0 );
        float frac2 = clamp( ( scaled_t - Width ) / Fuzz, 0.0, 1.0 );

        frac1 = frac1 * ( 1.0 - frac2 );
        frac1 = frac1 * frac1 * ( 3.0 - ( 2.0 * frac1 ) );
        vec3 finalColor = mix( BackColor, StripeColor, frac2 );
   
		FragColor = vec4( ads( finalColor ), 1.0 );
	} else if( ShaderType == 4 ) {

		float ss = fract( TexCoord.s * Scale );
		float tt = fract( TexCoord.t * Scale );
		if( ( ss > Threshold.s ) && ( tt > Threshold.t ) ) discard;

		FragColor = vec4( ads( vec3(1.0f, 1.0f, 1.0f ) ), 1.0 );
	}
}