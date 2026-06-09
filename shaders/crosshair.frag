#version 450

layout(location = 0) out vec4 out_color;

void main() {
    // White source color; inversion blend mode makes this always-visible against any background.
    out_color = vec4(1.0, 1.0, 1.0, 1.0);
}
