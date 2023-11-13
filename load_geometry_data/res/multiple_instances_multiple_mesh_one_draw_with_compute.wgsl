struct PartData {
  numVerticesMesh: u32,
  numFacesInPart: u32,
  meshOffset: u32,
  numMeshInstances: u32,
  firstInstance: u32,
  firstPart: u32,
};

struct InstanceData {
  x: f32,
  z: f32,
};

@group(0) @binding(0) var<storage, read_write> vertex_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> part_data: array<PartData>;
@group(0) @binding(3) var<storage, read> instance_data: array<InstanceData>;

const faces_per_part = 64u;

@compute @workgroup_size(64, 1, 1) fn computeSomething(
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
  @builtin(workgroup_id) id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  // calculate and populate in_vertex xyz data for Mesh instances
  let mesh_part = id.x;
  let num_mesh_parts = num_workgroups.x;
  let vertices_per_face = num_workgroups.z;
  let vertex_in_face = id.z;
  let face_in_part = local_invocation_id[0];
  // don't write to vertex_data if in_face index exceeds faces in mesh partition
  let instance = id.y;
  if (part_data[mesh_part].numMeshInstances > instance){
    if (part_data[mesh_part].numFacesInPart > face_in_part){
      let in_face = (mesh_part * faces_per_part) + face_in_part;
      let out_face = (((mesh_part - part_data[mesh_part].firstPart) * faces_per_part) + face_in_part);
      let in_vertex = (in_face * vertices_per_face) + vertex_in_face;
      let out_vertex = (out_face * vertices_per_face) + vertex_in_face;
      let num_vertices = part_data[mesh_part].numVerticesMesh;
      let num_screen_objects = num_workgroups.y;
      let first_instance = part_data[mesh_part].firstInstance;

      let x = instance_data[first_instance + instance].x;
      let z = instance_data[first_instance + instance].z;
      let instance_offset = num_vertices * instance;
      let mesh_offset = part_data[mesh_part].meshOffset;
      vertex_data[mesh_offset + instance_offset + out_vertex] = vec4<f32>(x, 0f, z, 0f) + mesh_data[in_vertex];
    }
  }

}