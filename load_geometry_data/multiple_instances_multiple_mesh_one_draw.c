#include "cglm/mat4.h"
#include "cglm/vec3.h"
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

// create a struct to hold the values for the uniforms
typedef struct Uniforms {
  mat4 matrix;
} Uniforms;

float degToRad(float d) { return d * M_PI / 180; };

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

  WGPUShaderModuleDescriptor shaderSource =
      load_wgsl(RESOURCE_DIR "shader.wgsl");
  WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &shaderSource);

  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
          .label = "2 attributes",
          .vertex =
              (WGPUVertexState){
                  .module = module,
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
                  .module = module,
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

#define numScreenObjects 10
#define numMeshes 3

  int i;
  int radius = 200;

  int fieldOfView = 100;
  float cameraAngle = 0;
  Uniforms uniformValues = {};
  const uint64_t uniformBufferSize = (16) * 4;

  int mesh_index = 0;
  int mesh_order[numScreenObjects];
  float *mesh_xz_coords[numMeshes];
  int numInstances[numMeshes];
  int mesh_instances[numMeshes];
  int mesh_xz_coords_size[numMeshes];
  // set instances to zero for all meshes
  for (i = 0; i < numMeshes; i++) {
    numInstances[i] = 0;
    mesh_instances[i] = 0;
  };
  // Generate mesh order placement
  printf("Generated preset mesh order:\n");
  for (i = 0; i < numScreenObjects; i++) {
    mesh_order[i] = mesh_index;
    printf("%i\n", mesh_order[i]);
    numInstances[mesh_order[i]]++; // increase instance count for this mesh
    mesh_index = mesh_index + 1 < numMeshes ? mesh_index + 1 : 0;
  }
  printf("-----------------\n");


  // set xz coords of ScreenObjects
  // calculate size of and allocate memorey for mesh_xz_coords
  for (i = 0; i < numMeshes; i++) {
    mesh_xz_coords_size[i] = numInstances[i] * sizeof(float) * 2;
    mesh_xz_coords[i] = (float *)malloc(mesh_xz_coords_size[i]);
  }
  printf("Generated F xz coordinates:\n");
  for (i = 0; i < numScreenObjects; i++) {
    float angle = ((float)i / numScreenObjects) * M_PI * 2;
    mesh_index = mesh_order[i];
    float *xz_coords = mesh_xz_coords[mesh_index];
    xz_coords[mesh_instances[mesh_index] * 2] = cosf(angle) * radius;
    xz_coords[mesh_instances[mesh_index] * 2 + 1] = sinf(angle) * radius;
    printf("%f %f\n", xz_coords[mesh_instances[mesh_index] * 2],
           xz_coords[mesh_instances[mesh_index] * 2 + 1]);
    mesh_instances[mesh_index]++;
  };
  printf("-----------------\n");

  // calculate xz_coords_size
  float *_vertexData[numMeshes];
  int _vertexDataSize[numMeshes];
  float *vertexData;
  int vertexDataSize;
  long _numVertices[numMeshes];
  char *ply_file_paths[numMeshes] = {
      RESOURCE_DIR "f_vertex_data_flipped_centered.ply", RESOURCE_DIR "B.ply",
      RESOURCE_DIR "b.ply"};

  WGPUBuffer uniformBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "uniforms",
                                      .size = uniformBufferSize,
                                      .usage = WGPUBufferUsage_Uniform |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});

  WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                  .label = "bind group for object",
                  .layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
                  .entries =
                      (WGPUBindGroupEntry[]){
                          {.binding = 0,
                           .buffer = uniformBuffer,
                           .size = uniformBufferSize,
                           .offset = 0},
                      },
                  .entryCount = 1});

  for (i = 0; i < numMeshes; i++) {
    // Open ply files, and read into seperate vertexData arrays
    get_vertex_data(ply_file_paths[i], &_vertexData[i], &_vertexDataSize[i],
                    &_numVertices[i]);
  }

  // calculate total numVertices
  uint32_t numVertices = 0;
  for (i = 0; i < numMeshes; i++) {
    numVertices += _numVertices[i] * numInstances[i];
  }

  // calculate and populate vertex xyz data for mesh instances
  vertexDataSize = numVertices * 4 * sizeof(float);
  vertexData = (float *)malloc(vertexDataSize);
  int meshoffset = 0;
  for (i = 0; i < numMeshes; i++) {
    int j = 0;
    int k = 0;
    float *vertecies = _vertexData[i];
    float *xz_coords = mesh_xz_coords[i];
    for (j = 0; j < numInstances[i]; j++) {
      int instance_offset = meshoffset + _numVertices[i] * j * 4;
      int xz_coords_offset = j * 2;

      for (k = 0; k < _numVertices[i]; k++) {
        int verticie_offset = k * 4;
        // add xz offsets and put into vertexData array
        glm_vec3_add((float[3]){xz_coords[xz_coords_offset], 0,
                                xz_coords[xz_coords_offset + 1]},
                     (float[3]){vertecies[verticie_offset],
                                vertecies[verticie_offset + 1],
                                vertecies[verticie_offset + 2]},
                     &vertexData[instance_offset + verticie_offset]);
        // copy color data
        vertexData[instance_offset + verticie_offset + 3] =
            vertecies[verticie_offset + 3];
      }
    }
    meshoffset += _numVertices[i] * 4 * numInstances[i];

    free(_vertexData[i]);
  };

  WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "vertex buffer vertices",
                                      .size = vertexDataSize,
                                      .usage = WGPUBufferUsage_Vertex |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});
  wgpuQueueWriteBuffer(queue, vertexBuffer, 0, vertexData,

                       vertexDataSize);

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

  WGPUTexture depthTexture = NULL;

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
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);

    int aspect = window_width / window_height;
    mat4 projection;
    glm_perspective(degToRad(fieldOfView), aspect, 1, 1000000, projection);

    // Position of the first F
    float fPosition[3] = {mesh_xz_coords[0][0], 0, mesh_xz_coords[0][1]};

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

    // upload the uniform values to the uniform buffer
    wgpuQueueWriteBuffer(queue, uniformBuffer, 0, &uniformValues,
                         sizeof(uniformValues));
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertexBuffer, 0,
                                         vertexDataSize);
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