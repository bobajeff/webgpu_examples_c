// WebGPU and C port of  'An introduction to Shader Art Coding' @ https://www.youtube.com/watch?v=f4s1h2YETNY & https://www.shadertoy.com/view/mtyGWy
struct Uniforms {
  iResolution: vec2<f32>,
  iTime: f32
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
@builtin(vertex_index) vertexIndex : u32
) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>( -1.0,  1.0),  // top left
        vec2<f32>(-1.0, -1.0),  // bottom left
        vec2<f32>( 1.0, -1.0),   // bottom right
        vec2<f32>( -1.0,  1.0),  // top left
        vec2<f32>( 1.0, -1.0),   // bottom right
        vec2<f32>(1.0, 1.0),  // top right
    );

    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = vec2<f32>();
    uv.x = position.x / uni.iResolution.x;
    uv.y = 1.0 - (position.y / uni.iResolution.y); // invert y
    uv = uv * 2.0 - 1.0;
    uv.x *= uni.iResolution.x / uni.iResolution.y;

    var d = length(uv);
    d = sin(d * 8.0 + uni.iTime) / 8.0;
    d = abs(d);

    d = smoothstep(0.0, 0.1, d);

    return vec4<f32>(d, d, d, 1.0);
}