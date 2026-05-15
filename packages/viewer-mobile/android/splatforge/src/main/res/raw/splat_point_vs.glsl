#version 310 es
// Mirror of `SplatPointSprite.metal::splat_point_vertex`.
layout(location=0) in vec3 a_position;
layout(location=1) in vec4 a_rotation;
layout(location=2) in vec3 a_scale;
layout(location=3) in float a_opacity;
layout(location=4) in vec3 a_color;

uniform mat4 u_viewProj;

out vec2 v_uv;
out vec3 v_color;
out float v_opacity;

const vec2 kQuad[4] = vec2[4](
    vec2(-1.0, -1.0), vec2(1.0, -1.0),
    vec2(-1.0,  1.0), vec2(1.0,  1.0)
);

void main() {
    vec4 clip = u_viewProj * vec4(a_position, 1.0);
    float pxRadius = clamp(a_scale.x * 200.0 / max(clip.w, 0.001), 1.0, 64.0);
    vec2 ndcOff = kQuad[gl_VertexID] * pxRadius / 800.0;
    gl_Position = vec4(clip.xy + ndcOff * clip.w, clip.z, clip.w);
    v_uv = kQuad[gl_VertexID];
    v_color = a_color;
    v_opacity = a_opacity;
}
