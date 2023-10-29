struct Uniforms {
  matrix: mat4x4<f32>,
};
 
struct Vertex {
  @location(0) position: vec4<f32>,
  @location(1) color: vec4<f32>,
};
 
struct VSOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> xz_coords: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> uni: Uniforms;
 
@vertex fn vs(vert: Vertex, @builtin(instance_index) instanceIndex : u32) -> VSOutput {
  var vsOut: VSOutput;
  let translation_mat = mat4x4<f32>(
    vec4<f32>(1.0, 0.0, 0.0, 0.0),
    vec4<f32>(0.0, 1.0, 0.0, 0.0),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(xz_coords[instanceIndex].x, 0.0, xz_coords[instanceIndex].y, 1.0));
  vsOut.position = uni.matrix * translation_mat * vert.position;
  vsOut.color = vert.color;
  return vsOut;
}
 
@fragment fn fs(vsOut: VSOutput) -> @location(0) vec4<f32> {
  return vsOut.color;
}