#version 450

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform sampler2DArray tex_array;

layout(location = 0) out vec4 out_color;

void main() {
    // uv.y encodes both V tile coord and layer: v_tile + layer * 256.0
    float layer = floor(frag_uv.y / 256.0);
    float v     = mod(frag_uv.y, 256.0);
    vec4 texel  = texture(tex_array, vec3(frag_uv.x, v, layer));

    vec3 light_dir   = normalize(vec3(0.6, 1.0, 0.4));
    float diffuse    = max(dot(normalize(frag_normal), light_dir), 0.0);
    float brightness = 0.25 + diffuse * 0.75;

    out_color = vec4(texel.rgb * brightness, texel.a);
}
