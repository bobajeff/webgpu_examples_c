@group(0) @binding(0) var<storage, read_write> vertexData: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> meshData: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> radius: u32;

const faces_per_part = 64u;

@compute @workgroup_size(64, 1, 1) fn computeSomething(
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>, @builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  // calculate and populate vertex xyz data for B instances
  let mesh_part = id.x;
  let num_mesh_parts = num_workgroups.x;
  let verticesPerFace = num_workgroups.z;
  let face_vertex = id.z;
  let part_face = local_invocation_id[0];
  let face = (mesh_part * faces_per_part) + part_face;
  let vertex = (face * verticesPerFace) + face_vertex;
  let PI = radians(180.0);
  let numVertices = num_mesh_parts * faces_per_part * verticesPerFace;
  let B_instance = id.y;
  let numBs = num_workgroups.y;

  let angle = (f32(B_instance) / f32(numBs)) * PI * 2f;
  let x = cos(angle) * f32(radius);
  let z = sin(angle) * f32(radius);
  let instance_offset = numVertices * B_instance;
  vertexData[instance_offset + vertex] = vec4<f32>(x, 0f, z, 0f) + meshData[vertex];

}