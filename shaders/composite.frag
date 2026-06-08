#version 450

layout(location = 0) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform sampler2D scene_tex;
layout(set = 0, binding = 1) uniform sampler2D ao_tex;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(scene_tex, frag_uv);
}
