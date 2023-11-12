struct PartData {
  numVerticesMesh: u32,
  num_faces_in_part: u32,
  mesh_offset: u32,
  numMeshInstances: u32,
  first_instance: u32,
  first_part: u32,
};

struct InstanceData {
  x: f32,
  z: f32,
};

@group(0) @binding(0) var<storage, read_write> vertexData: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> meshData: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> partData: array<PartData>;
@group(0) @binding(3) var<storage, read> instanceData: array<InstanceData>;

const faces_per_part = 64u;

@compute @workgroup_size(64, 1, 1) fn computeSomething(
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>, @builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  // calculate and populate in_vertex xyz data for Mesh instances
  let mesh_part = id.x;
  let num_mesh_parts = num_workgroups.x;
  let verticesPerFace = num_workgroups.z;
  let vertex_in_face = id.z;
  let face_in_part = local_invocation_id[0];
  // don't write to vertexData if in_face index exceeds faces in mesh partition
  let instance = id.y;
  if (partData[mesh_part].numMeshInstances > instance){
    if (partData[mesh_part].num_faces_in_part > face_in_part){
      let in_face = (mesh_part * faces_per_part) + face_in_part;
      let out_face = (((mesh_part - partData[mesh_part].first_part) * faces_per_part) + face_in_part);
      let in_vertex = (in_face * verticesPerFace) + vertex_in_face;
      let out_vertex = (out_face * verticesPerFace) + vertex_in_face;
      let numVertices = partData[mesh_part].numVerticesMesh;
      let numScreenObjects = num_workgroups.y;
      let first_instance = partData[mesh_part].first_instance;

      let x = instanceData[first_instance + instance].x;
      let z = instanceData[first_instance + instance].z;
      let instance_offset = numVertices * instance;
      let mesh_offset = partData[mesh_part].mesh_offset;
      vertexData[mesh_offset + instance_offset + out_vertex] = vec4<f32>(x, 0f, z, 0f) + meshData[in_vertex];
    }
  }

}