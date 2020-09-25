#version 410

in vec3 FrontColor;
in vec3 BackColor;
in vec2 TexCoord;

uniform float Fuzz = 0.5;
uniform float Scale = 10.0;
uniform float Width = 5;

layout(location = 0) out vec4 FragColor;

void main() {

	float scaledT = fract(TexCoord.x * Scale);
	float frac1 = clamp(scaledT / Fuzz, 0.0, 1.0);
	float frac2 = clamp((scaledT - Width) / Fuzz, 0.0, 1.0);

	frac1 = frac1 * (1.0 - frac2);
	frac1 = frac1 * frac1 * (3.0 - (2.0 * frac1));

	vec3 finalColor = mix(BackColor, FrontColor, frac1);
	finalColor = finalColor * BackColor + FrontColor;
	FragColor = vec4(finalColor, 1.0);
}