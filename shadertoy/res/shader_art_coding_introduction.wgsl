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

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.263,0.416,0.557);

    return a + b*cos( 6.28318*(c*t+d) );
}

@fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = vec2<f32>();
    uv.x = position.x / uni.iResolution.x;
    uv.y = 1.0 - (position.y / uni.iResolution.y); // invert y
    uv = uv * 2.0 - 1.0;
    uv.x *= uni.iResolution.x / uni.iResolution.y;
    let uv0 = uv;
    var finalColor = vec3<f32>(0.0);

    for (var i = 0; i < 4; i++) {

        uv = fract(uv * 1.5) - 0.5;

        var d = length(uv) * exp(-length(uv0));

        var col = palette(length(uv0) + f32(i) * 0.4 + uni.iTime * 0.4);

        d = sin(d * 8.0 + uni.iTime) / 8.0;
        d = abs(d);

        d = 0.01 / d;

        finalColor += col * d;
    }


    return vec4<f32>(finalColor, 1.0);
}