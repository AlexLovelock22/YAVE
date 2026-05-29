#version 450

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 light_dir = normalize(vec3(0.6, 1.0, 0.4));
    float diffuse = max(dot(normalize(frag_normal), light_dir), 0.0);
    float ambient = 0.25;
    float brightness = ambient + diffuse * (1.0 - ambient);
    out_color = vec4(vec3(0.55, 0.75, 0.95) * brightness, 1.0);
}
