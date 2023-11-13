#include "create_surface.h"
#include "framework.h"
#include "read_ply_file.h"
#include "webgpu.h"
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

typedef struct Uniforms {
  mat4 matrix;
} Uniforms;

typedef struct PartMap {
  u_int32_t numVerticesMesh;
  u_int32_t numFacesInPart;
  u_int32_t meshOffset;
  u_int32_t numMeshInstances;
  u_int32_t firstInstance;
  u_int32_t firstPart;
} PartMap;

typedef struct InstanceData {
  float x;
  float z;
} InstanceData;

float degToRad(int d) { return d * M_PI / 180; };

int main(int argc, char *argv[]) {
  srand(time(NULL)); // seed random number generator

  initializeLog();

  WGPUInstance instance =
      wgpuCreateInstance(&(WGPUInstanceDescriptor){.nextInChain = NULL});

  WGPUAdapter adapter;
  wgpuInstanceRequestAdapter(instance, NULL, request_adapter_callback,
                             (void *)&adapter);

  WGPUDevice device;
  wgpuAdapterRequestDevice(adapter, NULL, request_device_callback,
                           (void *)&device);

  WGPUQueue queue = wgpuDeviceGetQueue(device);

  wgpuDeviceSetUncapturedErrorCallback(device, handle_uncaptured_error, NULL);

  // Create GLFW Window and use as WebGPU surface
  if (!glfwInit()) {
    printf("Cannot initialize glfw");
    return 1;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(640, 480, "wgpu with glfw", NULL, NULL);

  if (!window) {
    printf("Cannot create window");
    return 1;
  }
  WGPUSurface surface = create_surface(instance, window);
  WGPUTextureFormat presentation_format =
      wgpuSurfaceGetPreferredFormat(surface, adapter);

  WGPUShaderModuleDescriptor compute_shader_source =
      load_wgsl(RESOURCE_DIR
                "multiple_instances_multiple_mesh_one_draw_with_compute.wgsl");
  WGPUShaderModule compute_module =
      wgpuDeviceCreateShaderModule(device, &compute_shader_source);

#define NUM_SCREEN_OBJECTS 10
#define NUM_MESHES 3
#define PARTSIZE 64
  uint32_t radius = 200;

  uint32_t num_vertices = 0;
  int mesh_bundle_size;
  float *mesh_bundle;

  u_int32_t num_mesh_instances[NUM_MESHES];
  memset(num_mesh_instances, 0, sizeof(u_int32_t) * NUM_MESHES);
  char *ply_file_paths[NUM_MESHES] = {
      RESOURCE_DIR "f_vertex_data_flipped_centered.ply", RESOURCE_DIR "B.ply",
      RESOURCE_DIR "b.ply"};
  int i, j;
  int mesh_index = 0;
  int mesh_order[NUM_SCREEN_OBJECTS];
  InstanceData instance_data[NUM_SCREEN_OBJECTS];
  int instance_data_size = sizeof(instance_data);

  // Generate mesh order placement
  for (i = 0; i < NUM_SCREEN_OBJECTS; i++) {
    mesh_order[i] = mesh_index;
    num_mesh_instances[mesh_index]++;
    mesh_index = mesh_index + 1 < NUM_MESHES ? mesh_index + 1 : 0;
  }

  // Calculate mesh partiton data
  u_int32_t total_mesh_parts = 0;
  int num_mesh_parts[NUM_MESHES];
  long raw_num_vertices_mesh[NUM_MESHES];
  u_int32_t remaining_faces[NUM_MESHES];
  float *raw_mesh_data[NUM_MESHES];
  int raw_mesh_data_size[NUM_MESHES];
  int mesh_data_size_with_padding[NUM_MESHES];
  int parts_offset[NUM_MESHES];
  int mesh_next_instance[NUM_MESHES];
  int highest_instance_count = 0;

  for (i = 0; i < NUM_MESHES; i++) {
    highest_instance_count = num_mesh_instances[i] > highest_instance_count
                                 ? num_mesh_instances[i]
                                 : highest_instance_count;
    parts_offset[i] = total_mesh_parts;
    // Open ply file, and read into vertexData
    get_vertex_data(ply_file_paths[i], &raw_mesh_data[i],
                    &raw_mesh_data_size[i], &raw_num_vertices_mesh[i]);
    int raw_num_face_mesh = raw_num_vertices_mesh[i] / 3;
    num_mesh_parts[i] = raw_num_face_mesh / PARTSIZE;
    remaining_faces[i] = raw_num_face_mesh % PARTSIZE;
    int num_face_mesh = num_mesh_parts[i] * PARTSIZE;
    if (raw_num_face_mesh > num_face_mesh) {
      num_mesh_parts[i]++;
    }
    total_mesh_parts += num_mesh_parts[i];
    long num_vertices_mesh_with_padding = num_mesh_parts[i] * PARTSIZE * 3;

    mesh_data_size_with_padding[i] =
        num_vertices_mesh_with_padding * 4 * sizeof(float);
  }

  // Set mesh partition data and add mesh to mesh "spritesheet"
  int part_map_size = total_mesh_parts * sizeof(PartMap);
  PartMap *part_map = (PartMap *)malloc(part_map_size);
  mesh_bundle_size = total_mesh_parts * PARTSIZE * 3 * 4 * sizeof(float);
  mesh_bundle = (float *)malloc(mesh_bundle_size);
  memset(mesh_bundle, 0, mesh_bundle_size);

  int mesh_data_offset = 0;
  int first_instance = 0;
  for (i = 0; i < NUM_MESHES; i++) {
    int first_part = parts_offset[i];
    for (j = 0; j < num_mesh_parts[i]; j++) {
      part_map[first_part + j].numVerticesMesh = raw_num_vertices_mesh[i];
      part_map[first_part + j].numFacesInPart = PARTSIZE;
      part_map[first_part + j].meshOffset = num_vertices;
      part_map[first_part + j].numMeshInstances = num_mesh_instances[i];
      part_map[first_part + j].firstInstance = first_instance;
      part_map[first_part + j].firstPart = first_part;
    }
    mesh_next_instance[i] = first_instance;
    // change faces in last part to the remaining faces
    int last_part = first_part + num_mesh_parts[i] - 1;
    part_map[last_part].numFacesInPart = remaining_faces[i];

    int num_vertices_all_instances =
        raw_num_vertices_mesh[i] * num_mesh_instances[i];
    num_vertices += num_vertices_all_instances;
    memcpy(&mesh_bundle[mesh_data_offset], raw_mesh_data[i],
           raw_mesh_data_size[i]);
    free(raw_mesh_data[i]);
    mesh_data_offset += mesh_data_size_with_padding[i] / sizeof(float);
    first_instance += num_mesh_instances[i];
  }

  // populate instance_data for Meshes
  for (i = 0; i < NUM_SCREEN_OBJECTS; i++) {
    float angle = ((float)i / NUM_SCREEN_OBJECTS) * M_PI * 2;
    instance_data[mesh_next_instance[mesh_order[i]]].x = cosf(angle) * radius;
    instance_data[mesh_next_instance[mesh_order[i]]].z = sinf(angle) * radius;
    mesh_next_instance[mesh_order[i]]++;
  };

  int vertex_data_size = num_vertices * 4 * sizeof(float);

  WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(
      device,
      &(WGPUComputePipelineDescriptor){
          .label = "compute pipeline",
          .compute = (WGPUProgrammableStageDescriptor){
              .module = compute_module, .entryPoint = "computeSomething"}});

  // create a buffer on the GPU to hold our computation
  // input and output
  WGPUBuffer mesh_bundle_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "read buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                  .size = mesh_bundle_size,
                  .mappedAtCreation = false,
              });
  WGPUBuffer vertex_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "vertex buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                  .size = vertex_data_size,
                  .mappedAtCreation = false,
              });

  WGPUBuffer part_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "part buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                  .size = part_map_size,
                  .mappedAtCreation = false,
              });
  WGPUBuffer instance_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "instance buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                  .size = instance_data_size,
                  .mappedAtCreation = false,
              });
  // Copy our input data to those buffers
  wgpuQueueWriteBuffer(queue, mesh_bundle_buffer, 0, mesh_bundle, mesh_bundle_size);
  wgpuQueueWriteBuffer(queue, part_buffer, 0, part_map, part_map_size);
  wgpuQueueWriteBuffer(queue, instance_buffer, 0, &instance_data[0],
                       instance_data_size);

  WGPUBindGroup compute_bind_group = wgpuDeviceCreateBindGroup(
      device,
      &(WGPUBindGroupDescriptor){
          .nextInChain = NULL,
          .layout = wgpuComputePipelineGetBindGroupLayout(compute_pipeline, 0),
          .entries = (WGPUBindGroupEntry[]){{.nextInChain = NULL,
                                             .binding = 0,
                                             .buffer = vertex_buffer,
                                             .offset = 0,
                                             .size = vertex_data_size},
                                            {.nextInChain = NULL,
                                             .binding = 1,
                                             .buffer = mesh_bundle_buffer,
                                             .offset = 0,
                                             .size = mesh_bundle_size},
                                            {.nextInChain = NULL,
                                             .binding = 2,
                                             .buffer = part_buffer,
                                             .offset = 0,
                                             .size = part_map_size},
                                            {.nextInChain = NULL,
                                             .binding = 3,
                                             .buffer = instance_buffer,
                                             .offset = 0,
                                             .size = instance_data_size}},
          .entryCount = 4, //**Important!**
      });

  WGPUShaderModuleDescriptor render_shader_source =
      load_wgsl(RESOURCE_DIR "shader.wgsl");
  WGPUShaderModule render_module =
      wgpuDeviceCreateShaderModule(device, &render_shader_source);

  WGPURenderPipeline render_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
          .label = "render pipeline",
          .vertex =
              (WGPUVertexState){
                  .module = render_module,
                  .entryPoint = "vs",
                  .bufferCount = 1,
                  .buffers =
                      (WGPUVertexBufferLayout[]){
                          {.arrayStride =
                               (4) *
                               4, // (3) floats 4 bytes each + one 4 byte color
                           .attributes =
                               (WGPUVertexAttribute[]){
                                   // position
                                   {.shaderLocation = 0,
                                    .offset = 0,
                                    .format = WGPUVertexFormat_Float32x3},
                                   {.shaderLocation = 1,
                                    .offset = 12,
                                    .format = WGPUVertexFormat_Unorm8x4},
                               },
                           .attributeCount = 2,
                           .stepMode = WGPUVertexStepMode_Vertex},
                      },
              },
          .primitive =
              (WGPUPrimitiveState){
                  .topology = WGPUPrimitiveTopology_TriangleList,
                  .stripIndexFormat = WGPUIndexFormat_Undefined,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode = WGPUCullMode_Back},
          .multisample =
              (WGPUMultisampleState){
                  .count = 1,
                  .mask = ~0,
                  .alphaToCoverageEnabled = false,
              },
          .fragment =
              &(WGPUFragmentState){
                  .module = render_module,
                  .entryPoint = "fs",
                  .targetCount = 1,
                  .targets =
                      &(WGPUColorTargetState){
                          .format = presentation_format,
                          .blend =
                              &(WGPUBlendState){
                                  .color =
                                      (WGPUBlendComponent){
                                          .srcFactor = WGPUBlendFactor_One,
                                          .dstFactor = WGPUBlendFactor_Zero,
                                          .operation = WGPUBlendOperation_Add,
                                      },
                                  .alpha =
                                      (WGPUBlendComponent){
                                          .srcFactor = WGPUBlendFactor_One,
                                          .dstFactor = WGPUBlendFactor_Zero,
                                          .operation = WGPUBlendOperation_Add,
                                      }},
                          .writeMask = WGPUColorWriteMask_All,
                      },
              },
          .depthStencil =
              &(WGPUDepthStencilState){
                  .nextInChain = NULL,
                  .format = WGPUTextureFormat_Depth24Plus,
                  .depthWriteEnabled = true,
                  .depthCompare = WGPUCompareFunction_Less,
                  .stencilFront =
                      (WGPUStencilFaceState){
                          .compare =
                              WGPUCompareFunction_Never, // magick value needed
                          .failOp = WGPUStencilOperation_Keep,
                          .depthFailOp = WGPUStencilOperation_Keep,
                          .passOp = WGPUStencilOperation_Keep,
                      },
                  .stencilBack =
                      (WGPUStencilFaceState){
                          .compare =
                              WGPUCompareFunction_Never, // magick value needed
                          .failOp = WGPUStencilOperation_Keep,
                          .depthFailOp = WGPUStencilOperation_Keep,
                          .passOp = WGPUStencilOperation_Keep,
                      },
                  .stencilReadMask = 0,
                  .stencilWriteMask = 0,
                  .depthBias = 0,
                  .depthBiasSlopeScale = 0.0,
                  .depthBiasClamp = 0.0},
      });

  WGPUBuffer uniform_buffer;
  WGPUBindGroup bind_group;
  Uniforms uniform_values = {};
  const uint64_t uniform_buffer_size = (16) * 4;

  // matrix
  uniform_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "uniforms",
                                      .size = uniform_buffer_size,
                                      .usage = WGPUBufferUsage_Uniform |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});

  bind_group = wgpuDeviceCreateBindGroup(
      device,
      &(WGPUBindGroupDescriptor){
          .label = "bind group for object",
          .layout = wgpuRenderPipelineGetBindGroupLayout(render_pipeline, 0),
          .entries =
              (WGPUBindGroupEntry[]){
                  {.binding = 0,
                   .buffer = uniform_buffer,
                   .size = uniform_buffer_size,
                   .offset = 0},
              },
          .entryCount = 1});

  WGPUSwapChainDescriptor config = (WGPUSwapChainDescriptor){
      .nextInChain =
          (const WGPUChainedStruct *)&(WGPUSwapChainDescriptorExtras){
              .chain =
                  (WGPUChainedStruct){
                      .next = NULL,
                      .sType = (WGPUSType)WGPUSType_SwapChainDescriptorExtras,
                  },
              .alphaMode = WGPUCompositeAlphaMode_Auto,
              .viewFormatCount = 0,
              .viewFormats = NULL,
          },
      .usage = WGPUTextureUsage_RenderAttachment,
      .format = presentation_format,
      .width = 0,
      .height = 0,
      .presentMode = WGPUPresentMode_Fifo,
  };

  glfwGetWindowSize(window, (int *)&config.width, (int *)&config.height);

  WGPUSwapChain swap_chain =
      wgpuDeviceCreateSwapChain(device, surface, &config);

  WGPUSupportedLimits limits = {};
  bool gotlimits = wgpuDeviceGetLimits(device, &limits);

  int field_of_view = 100;
  int camera_angle = 0;

  WGPUTexture depth_texture = NULL;

  // set xz coords of first Screen Object
  float b_xz_coords[NUM_SCREEN_OBJECTS][2];
  float angle = ((float)0 / NUM_SCREEN_OBJECTS) * M_PI * 2;
  b_xz_coords[0][0] = cosf(angle) * radius;
  b_xz_coords[0][1] = sinf(angle) * radius;

  while (!glfwWindowShouldClose(window)) {

    if (camera_angle < 360) {
      camera_angle += 1;
    } else {
      camera_angle = -360;
    }
    // printf("camera_angle: %i\n", camera_angle);

    WGPUTextureView view = NULL;

    for (int attempt = 0; attempt < 2; attempt++) {
      uint32_t prev_width = config.width;
      uint32_t prev_height = config.height;
      glfwGetWindowSize(window, (int *)&config.width, (int *)&config.height);

      if (prev_width != config.width || prev_height != config.height) {
        swap_chain = wgpuDeviceCreateSwapChain(device, surface, &config);
      }
      // Get the current texture from the swap_chain to use for rendering to by
      // the render pass
      view = wgpuSwapChainGetCurrentTextureView(swap_chain);
      if (attempt == 0 && !view) {
        printf("wgpuSwapChainGetCurrentTextureView() failed; trying to create "
               "a new swap chain...\n");
        config.width = 0;
        config.height = 0;
        continue;
      }

      break;
    }

    if (!view) {
      printf("Cannot acquire next swap chain texture\n");
      return 1;
    }

    // make a command encoder to start encoding commands
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        device, &(WGPUCommandEncoderDescriptor){.label = "our encoder"});

    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(
        encoder, &(WGPUComputePassDescriptor){
                     .label = "doubling compute pass",
                 });
    wgpuComputePassEncoderSetPipeline(compute_pass, compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, compute_bind_group, 0,
                                       NULL);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, total_mesh_parts,
                                             highest_instance_count, 3);
    wgpuComputePassEncoderEnd(compute_pass);

    int window_width, window_height;

    glfwGetWindowSize(window, (int *)&window_width, (int *)&window_height);

    window_width = (window_width < limits.limits.maxTextureDimension2D)
                       ? window_width
                       : limits.limits.maxTextureDimension2D;
    window_height = (window_height < limits.limits.maxTextureDimension2D)
                        ? window_height
                        : limits.limits.maxTextureDimension2D;

    window_width = (window_width > 1) ? window_width : 1;
    window_height = (window_height > 1) ? window_height : 1;

    // If we don't have a depth texture OR if its size is different
    // from the canvasTexture when make a new depth texture
    if (!depth_texture || wgpuTextureGetWidth(depth_texture) != window_width ||
        wgpuTextureGetHeight(depth_texture) != window_height) {
      if (depth_texture) {
        wgpuTextureDestroy(depth_texture);
      }
      depth_texture = wgpuDeviceCreateTexture(
          device, &(WGPUTextureDescriptor){
                      .nextInChain = NULL,
                      .label = NULL,
                      .usage = WGPUTextureUsage_RenderAttachment,
                      .dimension = WGPUTextureDimension_2D,
                      .size =
                          (WGPUExtent3D){
                              .width = window_width,
                              .height = window_height,
                              .depthOrArrayLayers = 1,
                          },
                      .format = WGPUTextureFormat_Depth24Plus,
                      .mipLevelCount = 1,
                      .sampleCount = 1,
                      .viewFormatCount = 0,
                      .viewFormats =
                          (WGPUTextureFormat[1]){WGPUTextureFormat_Undefined},
                  });
    }

    WGPUTextureView depth_stencil_attachment_view =
        wgpuTextureCreateView(depth_texture, NULL);

    WGPURenderPassDescriptor render_pass_descriptor = {
        .label = "our basic canvas renderPass",
        .colorAttachments =
            &(WGPURenderPassColorAttachment){
                .view = view, // texture from SwapChain
                .resolveTarget = NULL,
                .loadOp = WGPULoadOp_Clear,
                .storeOp = WGPUStoreOp_Store,
                .clearValue =
                    (WGPUColor){
                        .r = 0.3,
                        .g = 0.3,
                        .b = 0.3,
                        .a = 1.0,
                    },
            },
        .colorAttachmentCount = 1,
        .depthStencilAttachment =
            &(WGPURenderPassDepthStencilAttachment){
                .view = depth_stencil_attachment_view,
                .depthLoadOp = WGPULoadOp_Clear,
                .depthStoreOp = WGPUStoreOp_Store,
                .depthClearValue = 1.0,
                .depthReadOnly = false,
                .stencilLoadOp = WGPULoadOp_Clear,   // magick value needed
                .stencilStoreOp = WGPUStoreOp_Store, // magick value needed
                .stencilClearValue = 0,
                .stencilReadOnly = false,
            },
    };

    // make a render pass encoder to encode render specific commands
    WGPURenderPassEncoder pass =
        wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0,
                                         vertex_data_size);

    int aspect = window_width / window_height;
    mat4 projection;
    glm_perspective(degToRad(field_of_view), aspect, 1, 2000, projection);

    // Position of the first B
    float b_position[3] = {b_xz_coords[0][0], 0, b_xz_coords[0][1]};

    // Use matrix math to compute a position on a circle where
    // the camera is
    mat4 tmp_matrix, tmp_matrix_2;
    glm_mat4_identity(tmp_matrix);
    glm_rotate_y(tmp_matrix, degToRad(camera_angle), tmp_matrix_2);
    glm_translate_to(tmp_matrix_2, (float[3]){0, 0, radius * 1.5}, tmp_matrix);

    // Get the camera's position from the matrix we computed
    float eye[3] = {tmp_matrix[3][0], tmp_matrix[3][1], tmp_matrix[3][2]};

    float up[3] = {0, 1, 0};

    // Make a view matrix from the camera matrix.
    mat4 view_matrix;
    glm_lookat(eye, b_position, up, view_matrix);

    // combine the view and projection matrixes
    mat4 view_projection_matrix;
    glm_mat4_mul(projection, view_matrix, uniform_values.matrix);
    wgpuQueueWriteBuffer(queue, uniform_buffer, 0, &uniform_values,
                         sizeof(uniform_values));

    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, num_vertices, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);

    WGPUQueue queue = wgpuDeviceGetQueue(device);
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
        encoder, &(WGPUCommandBufferDescriptor){.label = NULL});
    wgpuQueueSubmit(queue, 1, &command_buffer);
    wgpuSwapChainPresent(swap_chain);

    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}