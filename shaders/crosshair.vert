#version 450

layout(push_constant) uniform Push {
    float aspect;
} push;

layout(location = 0) in vec2 pos;

void main() {
    // pos is in square-NDC-y space; divide x by aspect for equal pixel lengths.
    gl_Position = vec4(pos.x / push.aspect, pos.y, 0.0, 1.0);
}
