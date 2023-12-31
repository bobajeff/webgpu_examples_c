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

typedef struct PartData {
  u_int32_t numVerticesMesh;
  u_int32_t num_faces_in_part;
} PartData;

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
  WGPUTextureFormat presentationFormat =
      wgpuSurfaceGetPreferredFormat(surface, adapter);

  WGPUShaderModuleDescriptor computeShaderSource = load_wgsl(
      RESOURCE_DIR "compute_workgroup_size_64_no_excess_vertices.wgsl");
  WGPUShaderModule compute_module =
      wgpuDeviceCreateShaderModule(device, &computeShaderSource);

  #define numBs 10
  uint32_t radius = 200;
  // Open ply file, and read into vertexData
  float *rawMeshData;
  int rawMeshDataDataSize;
  long rawNumVerticesMesh;
  get_vertex_data(RESOURCE_DIR "B.ply", &rawMeshData, &rawMeshDataDataSize,
                  &rawNumVerticesMesh);

  // align mesh face count to multiple of 64
  #define PARTSIZE 64
  int rawNumFaceMesh = rawNumVerticesMesh / 3;
  printf("rawNumFaceMesh: %i\n", rawNumFaceMesh);
  int num_mesh_parts = rawNumFaceMesh / PARTSIZE;
  u_int32_t remaining_faces = rawNumFaceMesh % PARTSIZE;
  printf("num_mesh_parts: %i\n", num_mesh_parts);
  printf("remaining_faces: %i\n", remaining_faces);
  int numFaceMesh = num_mesh_parts * PARTSIZE;
  printf("numFaceMesh: %i\n", numFaceMesh);
  if (rawNumFaceMesh > numFaceMesh) {
    num_mesh_parts++;
    numFaceMesh = num_mesh_parts * PARTSIZE;
  } else {
    numFaceMesh = rawNumVerticesMesh;
  }
  printf("num_mesh_parts: %i\n", num_mesh_parts);
  printf("numFaceMesh: %i\n", numFaceMesh);
  long numVerticesMesh = numFaceMesh * 3;
  printf("numVerticesMesh: %li\n", numVerticesMesh);
  int i;
  int partDataSize = num_mesh_parts * sizeof(PartData);
  PartData *partData = (PartData *)malloc(partDataSize);
  for (i = 0; i < num_mesh_parts; i++) {
    partData[i].numVerticesMesh = rawNumVerticesMesh;
    partData[i].num_faces_in_part = PARTSIZE;
  }
  // change faces in last part to the remaining faces
  int last_part = num_mesh_parts - 1;
  partData[last_part].num_faces_in_part = remaining_faces;

  int meshDataSize = numVerticesMesh * 4 * sizeof(float);
  float *meshData = (float *)malloc(meshDataSize);

  // zero out excess values
  memset(meshData, 0, meshDataSize);
  memcpy(meshData, rawMeshData, rawMeshDataDataSize);
  free(rawMeshData);

  long numVertices = rawNumVerticesMesh * numBs;
  int vertexDataSize = numVertices * 4 * sizeof(float);
  float *vertexData = (float *)malloc(vertexDataSize);

  WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(
      device,
      &(WGPUComputePipelineDescriptor){
          .label = "compute pipeline",
          .compute = (WGPUProgrammableStageDescriptor){
              .module = compute_module, .entryPoint = "computeSomething"}});

  // create a buffer on the GPU to hold our computation
  // input and output
  WGPUBuffer meshBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "read buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                  .size = meshDataSize,
                  .mappedAtCreation = false,
              });
  WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "work buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                  .size = vertexDataSize,
                  .mappedAtCreation = false,
              });

  WGPUBuffer radiusBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "uniforms",
                                      .size = sizeof(radius),
                                      .usage = WGPUBufferUsage_Uniform |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});
  WGPUBuffer partBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                  .nextInChain = NULL,
                  .label = "read buffer",
                  .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                  .size = partDataSize,
                  .mappedAtCreation = false,
              });
  // Copy our input data to those buffers
  wgpuQueueWriteBuffer(queue, meshBuffer, 0, meshData, meshDataSize);
  wgpuQueueWriteBuffer(queue, radiusBuffer, 0, &radius, sizeof(radius));
  wgpuQueueWriteBuffer(queue, partBuffer, 0, partData, partDataSize);

  WGPUBindGroup computeBindGroup = wgpuDeviceCreateBindGroup(
      device,
      &(WGPUBindGroupDescriptor){
          .nextInChain = NULL,
          .layout = wgpuComputePipelineGetBindGroupLayout(compute_pipeline, 0),
          .entries = (WGPUBindGroupEntry[]){{.nextInChain = NULL,
                                             .binding = 0,
                                             .buffer = vertexBuffer,
                                             .offset = 0,
                                             .size = vertexDataSize},
                                            {.nextInChain = NULL,
                                             .binding = 1,
                                             .buffer = meshBuffer,
                                             .offset = 0,
                                             .size = meshDataSize},
                                            {.nextInChain = NULL,
                                             .binding = 2,
                                             .buffer = radiusBuffer,
                                             .offset = 0,
                                             .size = sizeof(radius)},
                                            {.nextInChain = NULL,
                                             .binding = 3,
                                             .buffer = partBuffer,
                                             .offset = 0,
                                             .size = partDataSize}},
          .entryCount = 4, //**Important!**
      });

  WGPUShaderModuleDescriptor renderShaderSource =
      load_wgsl(RESOURCE_DIR "shader.wgsl");
  WGPUShaderModule render_module =
      wgpuDeviceCreateShaderModule(device, &renderShaderSource);

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
                          .format = presentationFormat,
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

  WGPUBuffer uniformBuffer;
  WGPUBindGroup bindGroup;
  Uniforms uniformValues = {};
  const uint64_t uniformBufferSize = (16) * 4;

  // matrix
  uniformBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "uniforms",
                                      .size = uniformBufferSize,
                                      .usage = WGPUBufferUsage_Uniform |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});

  bindGroup = wgpuDeviceCreateBindGroup(
      device,
      &(WGPUBindGroupDescriptor){
          .label = "bind group for object",
          .layout = wgpuRenderPipelineGetBindGroupLayout(render_pipeline, 0),
          .entries =
              (WGPUBindGroupEntry[]){
                  {.binding = 0,
                   .buffer = uniformBuffer,
                   .size = uniformBufferSize,
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
      .format = presentationFormat,
      .width = 0,
      .height = 0,
      .presentMode = WGPUPresentMode_Fifo,
  };

  glfwGetWindowSize(window, (int *)&config.width, (int *)&config.height);

  WGPUSwapChain swapChain = wgpuDeviceCreateSwapChain(device, surface, &config);

  WGPUSupportedLimits limits = {};
  bool gotlimits = wgpuDeviceGetLimits(device, &limits);

  int fieldOfView = 100;
  int cameraAngle = 0;

  WGPUTexture depthTexture = NULL;

  // set xz coords of first B
  float b_xz_coords[numBs][2];
  float angle = ((float)0 / numBs) * M_PI * 2;
  b_xz_coords[0][0] = cosf(angle) * radius;
  b_xz_coords[0][1] = sinf(angle) * radius;

  while (!glfwWindowShouldClose(window)) {

    if (cameraAngle < 360) {
      cameraAngle += 1;
    } else {
      cameraAngle = -360;
    }
    // printf("cameraAngle: %i\n", cameraAngle);

    WGPUTextureView view = NULL;

    for (int attempt = 0; attempt < 2; attempt++) {
      uint32_t prevWidth = config.width;
      uint32_t prevHeight = config.height;
      glfwGetWindowSize(window, (int *)&config.width, (int *)&config.height);

      if (prevWidth != config.width || prevHeight != config.height) {
        swapChain = wgpuDeviceCreateSwapChain(device, surface, &config);
      }
      // Get the current texture from the swapChain to use for rendering to by
      // the render pass
      view = wgpuSwapChainGetCurrentTextureView(swapChain);
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
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, computeBindGroup, 0,
                                       NULL);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, num_mesh_parts,
                                             numBs, 3);
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
    if (!depthTexture || wgpuTextureGetWidth(depthTexture) != window_width ||
        wgpuTextureGetHeight(depthTexture) != window_height) {
      if (depthTexture) {
        wgpuTextureDestroy(depthTexture);
      }
      depthTexture = wgpuDeviceCreateTexture(
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

    WGPUTextureView depthStencilAttachment_view =
        wgpuTextureCreateView(depthTexture, NULL);

    WGPURenderPassDescriptor renderPassDescriptor = {
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
                .view = depthStencilAttachment_view,
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
        wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDescriptor);
    wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertexBuffer, 0,
                                         vertexDataSize);

    int aspect = window_width / window_height;
    mat4 projection;
    glm_perspective(degToRad(fieldOfView), aspect, 1, 2000, projection);

    // Position of the first B
    float fPosition[3] = {b_xz_coords[0][0], 0, b_xz_coords[0][1]};

    // Use matrix math to compute a position on a circle where
    // the camera is
    mat4 tmp_matrix, tmp_matrix_2;
    glm_mat4_identity(tmp_matrix);
    glm_rotate_y(tmp_matrix, degToRad(cameraAngle), tmp_matrix_2);
    glm_translate_to(tmp_matrix_2, (float[3]){0, 0, radius * 1.5}, tmp_matrix);

    // Get the camera's position from the matrix we computed
    float eye[3] = {tmp_matrix[3][0], tmp_matrix[3][1], tmp_matrix[3][2]};

    float up[3] = {0, 1, 0};

    // Make a view matrix from the camera matrix.
    mat4 viewMatrix;
    glm_lookat(eye, fPosition, up, viewMatrix);

    // combine the view and projection matrixes
    mat4 viewProjectionMatrix;
    glm_mat4_mul(projection, viewMatrix, uniformValues.matrix);
    wgpuQueueWriteBuffer(queue, uniformBuffer, 0, &uniformValues,
                         sizeof(uniformValues));

    wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, numVertices, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);

    WGPUQueue queue = wgpuDeviceGetQueue(device);
    WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(
        encoder, &(WGPUCommandBufferDescriptor){.label = NULL});
    wgpuQueueSubmit(queue, 1, &commandBuffer);
    wgpuSwapChainPresent(swapChain);

    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}