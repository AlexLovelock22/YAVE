#version 450

layout(location = 0) in vec2 frag_uv;

// Depth buffer from the geometry pass (depth-only aspect view)
layout(set = 0, binding = 0) uniform sampler2D depth_tex;

// proj and inv_proj matrices (with Vulkan Y-flip baked in)
layout(push_constant) uniform Push {
    mat4 proj;
    mat4 inv_proj;
} pc;

layout(location = 0) out float ao_out;

// Reconstruct view-space position from UV + depth
vec3 view_pos(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 v    = pc.inv_proj * clip;
    return v.xyz / v.w;
}

// Van der Corput radical inverse for Hammersley sampling
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

void main() {
    float depth = texture(depth_tex, frag_uv).r;
    // Sky pixels (nothing was rendered there) — fully lit, no AO
    if (depth >= 1.0) { ao_out = 1.0; return; }

    vec3 pos = view_pos(frag_uv, depth);

    // Reconstruct view-space normal from depth derivatives
    vec3 dx     = dFdx(pos);
    vec3 dy     = dFdy(pos);
    vec3 normal = normalize(cross(dx, dy));
    // Ensure the normal faces toward the camera (camera is at origin in view space)
    if (dot(normal, pos) > 0.0) normal = -normal;

    // Per-pixel random rotation (avoids banding without a noise texture)
    float angle   = fract(sin(dot(frag_uv, vec2(12.9898, 78.233))) * 43758.5453) * 6.28318;
    vec3 rand_vec = vec3(cos(angle), sin(angle), 0.0);
    vec3 tangent  = normalize(rand_vec - normal * dot(rand_vec, normal));
    if (length(tangent) < 0.001) tangent = normalize(cross(normal, vec3(0.0, 0.0, 1.0)));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    const float RADIUS    = 1.5;
    const float BIAS      = 0.05;
    const int   N_SAMPLES = 16;

    float occ = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        float fi = float(i);

        // Hammersley hemisphere sample (cosine-weighted)
        float u   = (fi + 0.5) / float(N_SAMPLES);
        float phi = acos(sqrt(1.0 - u));   // polar angle
        float theta = radical_inverse(uint(i)) * 6.28318; // azimuth

        // Hemisphere direction in tangent space (z > 0 = toward normal)
        float sin_phi = sin(phi);
        vec3  s       = vec3(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi));

        // Accelerating scale keeps more samples near the surface
        float scale    = mix(0.1, 1.0, fi * fi / float(N_SAMPLES * N_SAMPLES));
        vec3  s_pos    = pos + tbn * s * scale * RADIUS;

        // Project the sample back to screen UV
        vec4 proj_p = pc.proj * vec4(s_pos, 1.0);
        proj_p.xyz /= proj_p.w;
        vec2 s_uv = proj_p.xy * 0.5 + 0.5;

        // Skip samples that project outside the viewport
        if (any(lessThan(s_uv, vec2(0.0))) || any(greaterThan(s_uv, vec2(1.0)))) continue;

        float real_d = texture(depth_tex, s_uv).r;
        float real_z = view_pos(s_uv, real_d).z;

        // Range check: suppress false occlusion from distant geometry
        float range = smoothstep(0.0, 1.0, RADIUS / max(abs(pos.z - real_z), 0.001));

        // Occlusion: actual surface closer to camera than the sample position?
        // In view space: closer = larger z (less negative). real_z > s_pos.z + BIAS → occluded.
        occ += (real_z > s_pos.z + BIAS ? 1.0 : 0.0) * range;
    }

    // Amplify raw occlusion before output: multiply sensitivity so fewer occluded
    // samples are needed to produce a dark result. Tune this value (2.5) to taste.
    float raw = occ / float(N_SAMPLES);
    ao_out = 1.0 - clamp(raw * 2.5, 0.0, 1.0);
}
