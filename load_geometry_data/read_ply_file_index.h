#ifndef READ_PLY_VERTEX_XYZRGBA_INDEX
#define READ_PLY_VERTEX_XYZRGBA_INDEX
#include <stdlib.h>

int get_vertex_data(const char *ply_filepath, float **vertexData, int *vertexDataSize, long *numVertices, uint **indexData, int *indexDataSize, u_int32_t *indexCount);

#endif