#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "rply.h"

typedef struct vertex_pointers {
  float *vertexData;
  unsigned char *colorData;
  uint *indexData;
} vertex_pointers;

static int x_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->vertexData[instance_index * 4] = (float)ply_get_argument_value(argument);
  return 1;
}

static int y_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->vertexData[instance_index * 4 + 1] = (float)ply_get_argument_value(argument);
  return 1;
}

static int z_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->vertexData[instance_index * 4 + 2] = (float)ply_get_argument_value(argument);
  return 1;
}

static int r_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->colorData[instance_index * 16 + 12] = (unsigned char)ply_get_argument_value(argument);
  return 1;
}

static int g_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->colorData[instance_index * 16 + 12 + 1] =
      (unsigned char)ply_get_argument_value(argument);
  return 1;
}

static int b_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->colorData[instance_index * 16 + 12 + 2] =
      (unsigned char)ply_get_argument_value(argument);
  return 1;
}

static int a_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);

  v->colorData[instance_index * 16 + 12 + 3] =
      (unsigned char)ply_get_argument_value(argument);
  return 1;
}

static int indices_cb(p_ply_argument argument) {
  long instance_index;
  vertex_pointers *v;
  long value_index;
  long length;
  p_ply_property property;

  ply_get_argument_user_data(argument, (void **)&v, NULL);
  ply_get_argument_element(argument, NULL, &instance_index);
  ply_get_argument_property(argument, &property, &length, &value_index);
  // skip over the length value
  if (value_index == -1){
    uint length = (uint)ply_get_argument_value(argument);
    assert(length == 3); //sanity check that all faces are 3 vertices
    return 1;
  }
  v->indexData[instance_index * 3 + value_index] =
      (uint)ply_get_argument_value(argument);
  return 1;
}

int get_vertex_data(const char *ply_filepath, float **vertexData, int *vertexDataSize, long *numVertices, uint **indexData, int *indexDataSize, u_int32_t *indexCount) {
  // Open ply file, and read into vertexData
  p_ply ply = ply_open(ply_filepath, NULL, 0, NULL);
  if (!ply) {
    fprintf(stderr, "Can't open ply file\n");
    return 1;
  }
  if (!ply_read_header(ply)) {
    fprintf(stderr, "Can't read ply header\n");
    return 1;
  }
  vertex_pointers v;
  *numVertices = ply_set_read_cb(ply, "vertex", "x", x_cb, &v, 0);
  ply_set_read_cb(ply, "vertex", "y", y_cb, &v, 0);
  ply_set_read_cb(ply, "vertex", "z", z_cb, &v, 0);
  ply_set_read_cb(ply, "vertex", "red", r_cb, &v, 0);
  ply_set_read_cb(ply, "vertex", "green", g_cb, &v, 0);
  ply_set_read_cb(ply, "vertex", "blue", b_cb, &v, 0);
  ply_set_read_cb(ply, "vertex", "alpha", a_cb, &v, 0);
  long numfaces = ply_set_read_cb(ply, "face", "vertex_indices", indices_cb, &v, 0);
  *vertexDataSize = *numVertices * 4 * sizeof(float);
  *vertexData = (float *)malloc(*vertexDataSize);
  *indexCount = numfaces * 3;
  *indexDataSize = *indexCount * sizeof(uint);
  *indexData = (uint *)malloc(*indexDataSize);
  v.vertexData = *vertexData;
  v.colorData = (unsigned char *)*vertexData;
  v.indexData = *indexData;
  if (!ply_read(ply)) {
    fprintf(stderr, "Can't read element/property data from ply file\n");
    return 1;
  }
  ply_close(ply);
  return 1;
}