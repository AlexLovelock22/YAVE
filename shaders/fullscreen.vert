#version 450

layout(location = 0) out vec2 frag_uv;

void main() {
    // Full-screen triangle: vertices 0,1,2 cover the entire viewport without a VBO
    vec2 pos = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1)) * 4.0 - 1.0;
    frag_uv     = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
