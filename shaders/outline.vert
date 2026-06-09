#version 450

layout(push_constant) uniform Push {
    mat4 mvp;
} push;

layout(location = 0) in vec3 pos;

void main() {
    gl_Position = push.mvp * vec4(pos, 1.0);
}
