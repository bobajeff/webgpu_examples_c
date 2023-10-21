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

void aim(float * eye, float * target, float * up, mat4 dest) {

    vec3 tmp_vector;
    vec3 zAxis;
    glm_vec3_sub(target, eye, tmp_vector);
    glm_vec3_normalize_to(tmp_vector, zAxis);
    vec3 xAxis;
    glm_vec3_cross(up, zAxis, tmp_vector);
    glm_vec3_normalize_to(tmp_vector, xAxis);
    vec3 yAxis;
    glm_vec3_cross(zAxis, xAxis, tmp_vector);
    glm_vec3_normalize_to(tmp_vector, yAxis);

    dest[0][0]  = xAxis[0],  dest[0][1]  = xAxis[1],  dest[0][2]  = xAxis[2],  dest[0][3]  = 0,
    dest[1][0]  = yAxis[0],  dest[1][1]  = yAxis[1],  dest[1][2]  = yAxis[2],  dest[1][3]  = 0,
    dest[2][0]  = zAxis[0],  dest[2][1]  = zAxis[1],  dest[2][2]  = zAxis[2],  dest[2][3]  = 0,
    dest[3][0]  = eye[0],    dest[3][1]  = eye[1],    dest[3][2]  = eye[2],    dest[3][3]  = 1;

}

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
                  .cullMode = WGPUCullMode_Back
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

  
  const int numFs = 5 * 5 + 1;
  WGPUBuffer uniformBuffer[numFs];
  WGPUBindGroup bindGroup[numFs];
  Uniforms uniformValues = {};
  const uint64_t uniformBufferSize = (16) * 4;
  int i;
  for (i = 0; i < numFs; ++i) {
    // matrix
    uniformBuffer[i] = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){.label = "uniforms",
                                        .size = uniformBufferSize,
                                        .usage = WGPUBufferUsage_Uniform |
                                                WGPUBufferUsage_CopyDst,
                                        .mappedAtCreation = false});

    bindGroup[i] = wgpuDeviceCreateBindGroup(
        device,
        &(WGPUBindGroupDescriptor){
            .label = "bind group for object",
            .layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
            .entries = (WGPUBindGroupEntry[]){{.binding = 0,
                                                .buffer = uniformBuffer[i],
                                                .size = uniformBufferSize,
                                                .offset = 0},},
            .entryCount = 1});

  };


  // createFVertices()
  float positions[] = {
    // left column
     -50,  75,  15,
     -20,  75,  15,
     -50, -75,  15,
     -20, -75,  15,

    // top rung
     -20,  75,  15,
      50,  75,  15,
     -20,  45,  15,
      50,  45,  15,

    // middle rung
     -20,  15,  15,
      20,  15,  15,
     -20, -15,  15,
      20, -15,  15,

    // left column back
     -50,  75, -15,
     -20,  75, -15,
     -50, -75, -15,
     -20, -75, -15,

    // top rung back
     -20,  75, -15,
      50,  75, -15,
     -20,  45, -15,
      50,  45, -15,

    // middle rung back
     -20,  15, -15,
      20,  15, -15,
     -20, -15, -15,
      20, -15, -15,
  };

  u_int32_t indices[] = {
     0,  2,  1,    2,  3,  1,   // left column
     4,  6,  5,    6,  7,  5,   // top run
     8, 10,  9,   10, 11,  9,   // middle run

    12, 13, 14,   14, 13, 15,   // left column back
    16, 17, 18,   18, 17, 19,   // top run back
    20, 21, 22,   22, 21, 23,   // middle run back

     0,  5, 12,   12,  5, 17,   // top
     5,  7, 17,   17,  7, 19,   // top rung right
     6, 18,  7,   18, 19,  7,   // top rung bottom
     6,  8, 18,   18,  8, 20,   // between top and middle rung
     8,  9, 20,   20,  9, 21,   // middle rung top
     9, 11, 21,   21, 11, 23,   // middle rung right
    10, 22, 11,   22, 23, 11,   // middle rung bottom
    10,  3, 22,   22,  3, 15,   // stem right
     2, 14,  3,   14, 15,  3,   // bottom
     0, 12,  2,   12, 14,  2,   // left
  };

  unsigned char quadColors[] = {
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
  unsigned char * colorData = (unsigned char *)vertexData;
  for (i = 0; i < indices_length; i++){
      int positionNdx = indices[i] * 3;
      vertexData[i * 4] = positions[positionNdx];
      vertexData[i * 4 + 1] = positions[positionNdx + 1];
      vertexData[i * 4 + 2] = positions[positionNdx + 2];
  
      int quadNdx = (i / 6 | 0) * 3;
      colorData[i * 16 + 12] = quadColors[quadNdx];
      colorData[i * 16 + 12 + 1] = quadColors[quadNdx + 1];
      colorData[i * 16 + 12 + 2] = quadColors[quadNdx + 2];
      colorData[i * 16 + 12 + 3] = (unsigned char)255;
  }

  const WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(
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
  
  int radius = 200;

  float settings_target[3] = {0, 200, 300};
  int targetAngle = 0;
  int target_forward = 1;

  WGPUTexture depthTexture = NULL;
  
  while (!glfwWindowShouldClose(window)) {
    if (targetAngle < 360){
        targetAngle += 1;
    } else {
        targetAngle = -360;
    }
    printf("targetAngle: %i\n", targetAngle);

    if (target_forward){
        if (settings_target[1] < 300){
            settings_target[1] += 1;
        } else {
            target_forward = 0;
        }
    } else {
        if (settings_target[1] > -100){
            settings_target[1] -= 1;
        } else {
            target_forward = 1;
        }
    }
    printf("target.y %f\n", settings_target[1]);;


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

    
    int aspect = window_width / window_height;
    // update target X,Z based on angle
    settings_target[0] = cos(degToRad(targetAngle)) * radius;
    settings_target[2] = sin(degToRad(targetAngle)) * radius;

    mat4 projection;
    glm_perspective(degToRad(60), aspect, 1, 2000, projection);

    // Get the camera's position from the matrix we computed
    float eye[3] = {-500, 300, -500};
    float target[3] = {0, -100, 0};
    float up[3] = {0, 1, 0};

    // Make a view matrix from the camera matrix.
    mat4 viewMatrix;
    glm_lookat(eye, target, up, viewMatrix);

    // combine the view and projection matrixes
    mat4 viewProjectionMatrix;
    glm_mat4_mul(projection, viewMatrix, viewProjectionMatrix);
    for (i = 0; i < numFs; ++i) {
        int deep = 5;
        int across = 5;
        if (i < 25) {
            // compute grid positions
            float gridX = i % across;
            float gridZ = i / across | 0;

            // compute 0 to 1 positions
            float u = gridX / (across - 1);
            float v = gridZ / (deep - 1);

            // center and spread out
            float x = (u - 0.5) * across * 150;
            float z = (v - 0.5) * deep * 150;

            // aim this F from it's position toward the target F
            mat4 aimMatrix;
            aim((float[3]){x, 0, z}, settings_target, up, aimMatrix);
            glm_mat4_mul(viewProjectionMatrix, aimMatrix, uniformValues.matrix);
        } else {
            glm_translate_to(viewProjectionMatrix, (float[3]){settings_target[0], settings_target[1], settings_target[2]}, uniformValues.matrix);
        }

        // upload the uniform values to the uniform buffer
        wgpuQueueWriteBuffer(queue, uniformBuffer[i], 0, &uniformValues,
                            sizeof(uniformValues));

        wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup[i], 0, NULL);
        wgpuRenderPassEncoderDraw(pass, numVertices, 1, 0, 0);
    }
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