#version 450

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;

layout(location = 0) out vec4 out_color;

void main() {
    int block = int(round(frag_uv.x));
    vec3 base;
    if      (block == 1) base = vec3(0.42, 0.27, 0.12);  // DIRT:  dark brown
    else if (block == 4) base = vec3(0.10, 0.38, 0.78);  // WATER: deep blue
    else                 base = vec3(0.40, 0.40, 0.42);  // STONE: gray

    vec3 light_dir = normalize(vec3(0.6, 1.0, 0.4));
    float diffuse = max(dot(normalize(frag_normal), light_dir), 0.0);
    float ambient = 0.25;
    float brightness = ambient + diffuse * (1.0 - ambient);
    out_color = vec4(base * brightness, 1.0);
}
