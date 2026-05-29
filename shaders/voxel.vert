#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;

void main() {
    gl_Position = pc.mvp * vec4(in_pos, 1.0);
    frag_normal = in_normal;
    frag_uv = in_uv;
}
