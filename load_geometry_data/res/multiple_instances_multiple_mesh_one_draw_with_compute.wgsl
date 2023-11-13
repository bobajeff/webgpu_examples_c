struct PartMap {
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
@group(0) @binding(1) var<storage, read> mesh_bundle: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> part_map: array<PartMap>;
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
  let instance = id.y;
  // don't write to vertex_data if instance exceeds instances of mesh
  if (part_map[mesh_part].numMeshInstances > instance){
    // ' ' if face exceeds faces in partition
    if (part_map[mesh_part].numFacesInPart > face_in_part){
      let in_face = (mesh_part * faces_per_part) + face_in_part;
      let out_face = (((mesh_part - part_map[mesh_part].firstPart) * faces_per_part) + face_in_part);
      let in_vertex = (in_face * vertices_per_face) + vertex_in_face;
      let out_vertex = (out_face * vertices_per_face) + vertex_in_face;
      let num_vertices = part_map[mesh_part].numVerticesMesh;
      let first_instance = part_map[mesh_part].firstInstance;

      let x = instance_data[first_instance + instance].x;
      let z = instance_data[first_instance + instance].z;
      let instance_offset = num_vertices * instance;
      let mesh_offset = part_map[mesh_part].meshOffset;
      vertex_data[mesh_offset + instance_offset + out_vertex] = vec4<f32>(x, 0f, z, 0f) + mesh_bundle[in_vertex];
    }
  }

}