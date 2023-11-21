struct PartMap {
  meshMapIndex: u32,
  numFacesInPart: u32,
};

struct MeshMap {
  numMeshVertices: u32,
  outMeshVertexOffset: u32,
  numMeshInstances: u32,
  meshInstanceOffset: u32,
  meshPartsOffset: u32,
};

struct InstanceData {
  x: f32,
  z: f32,
};

@group(0) @binding(0) var<storage, read_write> vertex_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_bundle: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> part_maps: array<PartMap>;
@group(0) @binding(3) var<storage, read> instance_data: array<InstanceData>;
@group(0) @binding(4) var<storage, read> mesh_maps: array<MeshMap>;

const faces_per_part = 64u;

@compute @workgroup_size(64, 1, 1) fn computeSomething(
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
  @builtin(workgroup_id) id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  // calculate and populate in_vertex xyz data for Mesh instances
  let bundle_part = id.x;
  let vertices_per_face = num_workgroups.z; // 3
  let vertex_in_face = id.z;
  let face_in_part = local_invocation_id[0];
  let instance = id.y;
  let part_map = part_maps[bundle_part];
  let mesh_map = mesh_maps[part_map.meshMapIndex];

  // don't write to vertex_data if instance exceeds instances of mesh
  if (mesh_map.numMeshInstances > instance){
    // ' ' if face exceeds faces in partition
    if (part_map.numFacesInPart > face_in_part){
      let bundle_face = (bundle_part * faces_per_part) + face_in_part;
      let bundle_vertex = (bundle_face * vertices_per_face) + vertex_in_face;

      let mesh_instance_offset = mesh_map.meshInstanceOffset;
      let x = instance_data[mesh_instance_offset + instance].x;
      let z = instance_data[mesh_instance_offset + instance].z;

      let num_mesh_vertices = mesh_map.numMeshVertices;
      let out_instance_vertex_offset = num_mesh_vertices * instance;
      let out_mesh_vertex_offset = mesh_map.outMeshVertexOffset;
      let out_part = bundle_part - mesh_map.meshPartsOffset;
      let out_face = ((out_part * faces_per_part) + face_in_part);
      let out_vertex = (out_face * vertices_per_face) + vertex_in_face;
      vertex_data[out_mesh_vertex_offset + out_instance_vertex_offset + out_vertex] = vec4<f32>(x, 0f, z, 0f) + mesh_bundle[bundle_vertex];
    }
  }

}