@group(0) @binding(0) var<storage, read_write> vertexData: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> meshData: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> radius: u32;

@compute @workgroup_size(1) fn computeSomething(
  @builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  // calculate and populate vertex xyz data for F instances
  let PI = radians(180.0);
  let vertex = id.x;
  let numVertices = num_workgroups.x;
  let F_instance = id.y;
  let numFs = num_workgroups.y;

  let angle = (f32(F_instance) / f32(numFs)) * PI * 2f;
  let x = cos(angle) * f32(radius);
  let z = sin(angle) * f32(radius);
  let instance_offset = numVertices * F_instance;
  vertexData[instance_offset + vertex] = vec4<f32>(x, 0f, z, 0f) + meshData[vertex];
}