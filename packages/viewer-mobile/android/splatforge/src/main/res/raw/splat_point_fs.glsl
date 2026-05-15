#version 310 es
precision mediump float;
in vec2 v_uv;
in vec3 v_color;
in float v_opacity;
out vec4 fragColor;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;
    float a = exp(-4.0 * r2) * v_opacity;
    fragColor = vec4(v_color * a, a);
}
