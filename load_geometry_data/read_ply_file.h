#ifndef READ_PLY_VERTEX_XYZRGB
#define READ_PLY_VERTEX_XYZRGB


int get_vertex_data(const char *ply_filepath, float **vertexData, int *vertexDataSize, long *numVertices);

#endif