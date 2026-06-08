#version 450

layout(location = 0) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform sampler2D ao_tex;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;

layout(location = 0) out float ao_out;

void main() {
    vec2  texel    = 1.0 / vec2(textureSize(ao_tex, 0));
    float center_d = texture(depth_tex, frag_uv).r;

    float total = 0.0;
    float w_sum = 0.0;

    // 5x5 bilateral Gaussian blur (sigma=1.5).
    // depth weighting preserves hard block edges while smoothing grain on flat surfaces.
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2  uv      = frag_uv + vec2(float(x), float(y)) * texel;
            float d       = texture(depth_tex, uv).r;
            float spatial = exp(-0.5 * float(x*x + y*y) / 2.25); // sigma = 1.5
            float range   = 1.0 - smoothstep(0.0, 0.001, abs(d - center_d));
            float w       = spatial * range;
            total        += texture(ao_tex, uv).r * w;
            w_sum        += w;
        }
    }

    ao_out = total / max(w_sum, 0.001);
}
