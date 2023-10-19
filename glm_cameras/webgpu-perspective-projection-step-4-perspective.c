#include "cglm/affine.h"
#include "create_surface.h"
#include "framework.h"
#include "webgpu.h"
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <cglm/cglm.h>


// create a struct to hold the values for the uniforms
typedef struct Uniforms {
  mat4 matrix;
} Uniforms;

float degToRad(int d){ return d * M_PI / 180;};

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

  WGPUShaderModuleDescriptor shaderSource = load_wgsl(
      RESOURCE_DIR "shader.wgsl");
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
                          {.arrayStride = (4) * 4, // (3) floats 4 bytes each + one 4 byte color
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
                         },},
          .primitive =
              (WGPUPrimitiveState){
                  .topology = WGPUPrimitiveTopology_TriangleList,
                  .stripIndexFormat = WGPUIndexFormat_Undefined,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode = WGPUCullMode_Front // note: uncommon setting. See article
                  },
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
          .depthStencil = &(WGPUDepthStencilState){
            .nextInChain = NULL,
            .format = WGPUTextureFormat_Depth24Plus,
            .depthWriteEnabled = true,
            .depthCompare = WGPUCompareFunction_Less,
            .stencilFront = (WGPUStencilFaceState){
                .compare = WGPUCompareFunction_Never, // magick value needed 
                .failOp = WGPUStencilOperation_Keep,
                .depthFailOp = WGPUStencilOperation_Keep,
                .passOp = WGPUStencilOperation_Keep,
            },
            .stencilBack = (WGPUStencilFaceState){
                .compare = WGPUCompareFunction_Never, // magick value needed 
                .failOp = WGPUStencilOperation_Keep,
                .depthFailOp = WGPUStencilOperation_Keep,
                .passOp = WGPUStencilOperation_Keep,
            },
            .stencilReadMask = 0,
            .stencilWriteMask = 0,
            .depthBias = 0,
            .depthBiasSlopeScale = 0.0,
            .depthBiasClamp = 0.0
          },
      });

  Uniforms uniformValues = {};

  // matrix
  const uint64_t uniformBufferSize = (16) * 4;
  const WGPUBuffer uniformBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "uniforms",
                                      .size = uniformBufferSize,
                                      .usage = WGPUBufferUsage_Uniform |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});

  // createFVertices()
  float positions[] = {
    // left column
    0, 0, 0,
    30, 0, 0,
    0, 150, 0,
    30, 150, 0,

    // top rung
    30, 0, 0,
    100, 0, 0,
    30, 30, 0,
    100, 30, 0,

    // middle rung
    30, 60, 0,
    70, 60, 0,
    30, 90, 0,
    70, 90, 0,

    // left column back
    0, 0, 30,
    30, 0, 30,
    0, 150, 30,
    30, 150, 30,

    // top rung back
    30, 0, 30,
    100, 0, 30,
    30, 30, 30,
    100, 30, 30,

    // middle rung back
    30, 60, 30,
    70, 60, 30,
    30, 90, 30,
    70, 90, 30,
  };

  u_int32_t indices[] = {
    0,  1,  2,    2,  1,  3,  // left column
    4,  5,  6,    6,  5,  7,  // top rung
    8,  9, 10,   10,  9, 11,  // middle rung

    12, 14, 13,   14, 15, 13,  // left column back
    16, 18, 17,   18, 19, 17,  // top run back
    20, 22, 21,   22, 23, 21,  // middle run back

     0, 12,  5,   12, 17,  5,   // top
     5, 17,  7,   17, 19,  7,   // top rung right
     6,  7, 18,   18,  7, 19,   // top rung bottom
     6, 18,  8,   18, 20,  8,   // between top and middle rung
     8, 20,  9,   20, 21,  9,   // middle rung top
     9, 21, 11,   21, 23, 11,   // middle rung right
    10, 11, 22,   22, 11, 23,   // middle rung bottom
    10, 22,  3,   22, 15,  3,   // stem right
     2,  3, 14,   14,  3, 15,   // bottom
     0,  2, 12,   12,  2, 14,   // left
  };

  char quadColors[] = {
      200,  70, 120,  // left column front
      200,  70, 120,  // top rung front
      200,  70, 120,  // middle rung front
       80,  70, 200,  // left column back
       80,  70, 200,  // top rung back
       80,  70, 200,  // middle rung back
       70, 200, 210,  // top
      160, 160, 220,  // left side
       90, 130, 110,  // bottom
      200, 200,  70,  // top rung right
      210, 100,  70,  // under top rung
      210, 160,  70,  // between top rung and middle
       70, 180, 210,  // top of middle rung
      100,  70, 210,  // right of middle rung
       76, 210, 100,  // bottom of middle rung.
      140, 210,  80,  // right of bottom
  };

  int indices_length = sizeof(indices) / sizeof(u_int32_t);
  int numVertices = indices_length;
  int vertexDataSize = indices_length * 4 * sizeof(float);
  float *vertexData = (float *)malloc(vertexDataSize);
  char * colorData = (char *)vertexData;
  int i;
  for (i = 0; i < indices_length; i++){
      int positionNdx = indices[i] * 3;
      vertexData[i * 4] = positions[positionNdx];
      vertexData[i * 4 + 1] = positions[positionNdx + 1];
      vertexData[i * 4 + 2] = positions[positionNdx + 2];
  
      int quadNdx = (i / 6 | 0) * 3;
      colorData[i * 16 + 12] = quadColors[quadNdx];
      colorData[i * 16 + 12 + 1] = quadColors[quadNdx + 1];
      colorData[i * 16 + 12 + 2] = quadColors[quadNdx + 2];
      colorData[i * 16 + 12 + 3] = (char)225;
  }

  const WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){.label = "vertex buffer vertices",
                                      .size = vertexDataSize,
                                      .usage = WGPUBufferUsage_Vertex |
                                               WGPUBufferUsage_CopyDst,
                                      .mappedAtCreation = false});
  wgpuQueueWriteBuffer(queue, vertexBuffer, 0, vertexData,

                       vertexDataSize);

  WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(
      device,
      &(WGPUBindGroupDescriptor){
          .label = "bind group for object",
          .layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
          .entries = (WGPUBindGroupEntry[]){{.binding = 0,
                                              .buffer = uniformBuffer,
                                              .size = uniformBufferSize,
                                              .offset = 0},},
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
  
  int __window_width;
  int __window_height;
  glfwGetWindowSize(window, (int *)&__window_width, (int *)&__window_height);
  #define NUM_OF_PRESETS 10
  float translation[NUM_OF_PRESETS][3] = {
    {-65, 0.0, -120.0},
    {200.0,300.0, 0},
    {500.0,40.0, 0},
    {700.0,1000.0, 0},
    {300.0,200.0, 0},
    {40.0,500.0, 0},
    {1000.0,700.0, 0},
    {80.0,20.0, 0},
    {10.0,250.0, 0},
    {1000.0,1000.0, 0},
  };

  float rotation[3] = {220,25,325};
  int rotation_axis = 0;
  int preset = 0;
  int count = 0;

  float scale[3] = {1.0,1.0,1.0};
  int scale_forward = 1;
  int scale_axis = 0;

  int fieldOfView = 100;
  
  WGPUTexture depthTexture = NULL;
  
  while (!glfwWindowShouldClose(window)) {
    // change translation preset after 250 loops (because making a GUI is not as easy outside of the web)
    if (count < 250) 
    {
        count++;
    }
    else {
        scale_axis = scale_axis < 2 ? scale_axis + 1 : 0; //switch scale axis every 250 loop
        rotation_axis = rotation_axis < 2 ? rotation_axis + 1 : 0; //switch scale axis every 250 loop
        count = 0;
        // if (preset < NUM_OF_PRESETS + 1){
        //     preset++;
        // }
        // else {
        //     preset = 0;
        // }
    }
    printf("translation: %f,%f,%f\n", translation[preset][0],translation[preset][1], translation[preset][2]);
    // if (rotation[rotation_axis] < 360){
    //     rotation[rotation_axis] += 1;
    // } else {
    //     rotation[rotation_axis] = -360;
    // }
    printf("rotation: %f,%f,%f°\n", rotation[0], rotation[1], rotation[2]);

    // if (scale_forward){
    //     if (scale[scale_axis] < 5){
    //         scale[scale_axis] += 0.1;
    //     } else {
    //         scale_forward = 0;
    //     }
    // } else {
    //     if (scale[scale_axis] > -5){
    //         scale[scale_axis] -= 0.1;
    //     } else {
    //         scale_forward = 1;
    //     }
    // }
    printf("scale: %f,%f,%f\n", scale[0],scale[1], scale[2]);
    if (fieldOfView < 179){
        fieldOfView += 1;
    } else {
        fieldOfView = 1;
    }
    printf("fieldOfView: %i\n", fieldOfView);


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
    if (!depthTexture ||
        wgpuTextureGetWidth(depthTexture) != window_width ||
        wgpuTextureGetHeight(depthTexture) != window_height) {
      if (depthTexture) {
        wgpuTextureDestroy(depthTexture);
      }
      depthTexture = wgpuDeviceCreateTexture(device, &(WGPUTextureDescriptor){
        .nextInChain = NULL,
        .label = NULL,
        .usage = WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D,
        .size = (WGPUExtent3D){
            .width = window_width,
            .height = window_height,
            .depthOrArrayLayers = 1,
        },
        .format = WGPUTextureFormat_Depth24Plus,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 0,
        .viewFormats = (WGPUTextureFormat[1]){WGPUTextureFormat_Undefined},
      });
    }
   
   WGPUTextureView depthStencilAttachment_view = wgpuTextureCreateView(depthTexture, NULL);

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
        .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
            .view = depthStencilAttachment_view,
            .depthLoadOp = WGPULoadOp_Clear,
            .depthStoreOp = WGPUStoreOp_Store,
            .depthClearValue = 1.0,
            .depthReadOnly = false,
            .stencilLoadOp = WGPULoadOp_Clear, // magick value needed 
            .stencilStoreOp = WGPUStoreOp_Store, // magick value needed 
            .stencilClearValue = 0,
            .stencilReadOnly = false,
        },
    };

    // make a render pass encoder to encode render specific commands
    WGPURenderPassEncoder pass =
        wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDescriptor);
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertexBuffer, 0,
                                         vertexDataSize);

    mat4 tmp_matrix, tmp_matrix_2, tmp_matrix_3;
    int aspect = window_width / window_height;

    mat4 projection;
    glm_perspective(degToRad(fieldOfView), aspect, 1, 200, tmp_matrix);
    
    glm_translate_to(tmp_matrix, (float[3]){translation[preset][0], translation[preset][1], translation[preset][2]}, tmp_matrix_2);
    glm_rotate_x(tmp_matrix_2, degToRad(rotation[0]), tmp_matrix);
    glm_rotate_y(tmp_matrix, degToRad(rotation[1]), tmp_matrix_2);
    glm_rotate_z(tmp_matrix_2, degToRad(rotation[2]), tmp_matrix);
    glm_scale_to(tmp_matrix, (float[3]){scale[0], scale[1], scale[2]}, uniformValues.matrix);

    // upload the uniform values to the uniform buffer
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