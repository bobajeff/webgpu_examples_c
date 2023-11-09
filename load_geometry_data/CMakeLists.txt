cmake_minimum_required(VERSION 3.0...3.25)
project(
	Geometry_Data # name of the project, which will also be the name of the visual studio solution if you use it
	VERSION 0.1.0 # any version number
	LANGUAGES CXX C # programming languages used by the project
)
add_compile_definitions(RESOURCE_DIR="${CMAKE_SOURCE_DIR}/load_geometry_data/res/")

#--------------READ_PLY_FILE
add_library(read_ply_file
read_ply_file.c
)
target_link_libraries(read_ply_file PRIVATE rply)

#--------------USE_PLY_DATA
add_executable(use_ply_data
use_ply_data.c
)
target_link_libraries(use_ply_data PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_INSTANCES_OF_MESH
add_executable(multiple_instances_of_mesh
multiple_instances_of_mesh.c
)
target_link_libraries(multiple_instances_of_mesh PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_INSTANCES_USING_INSTANCECOUNT
add_executable(multiple_instances_using_instanceCount
multiple_instances_using_instanceCount.c
)
target_link_libraries(multiple_instances_using_instanceCount PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_OF_SEPERATE_MESHES
add_executable(multiple_of_seperate_meshes
multiple_of_seperate_meshes.c
)
target_link_libraries(multiple_of_seperate_meshes PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_INSTANCES_USING_INSTANCECOUNT_WITH_MULTIPLE_MESHES
add_executable(multiple_instances_using_instanceCount_with_multiple_meshes
multiple_instances_using_instanceCount_with_multiple_meshes.c
)
target_link_libraries(multiple_instances_using_instanceCount_with_multiple_meshes PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_INSTANCES_USING_INSTANCECOUNT_AND_STEPMODEINSTANCE
add_executable(multiple_instances_using_instanceCount_and_StepModeInstance
multiple_instances_using_instanceCount_and_StepModeInstance.c
)
target_link_libraries(multiple_instances_using_instanceCount_and_StepModeInstance PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_INSTANCES_USING_INSTANCECOUNT_FIRSTVERTEX_FIRSTINSTANCE
add_executable(multiple_instances_using_instanceCount_firstVertex_firstInstance
multiple_instances_using_instanceCount_firstVertex_firstInstance.c
)
target_link_libraries(multiple_instances_using_instanceCount_firstVertex_firstInstance PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------READ_PLY_FILE_INDEX
add_library(read_ply_file_index
read_ply_file_index.c
)
target_link_libraries(read_ply_file_index PRIVATE rply)

#--------------USE_PLY_DATA_DRAWINDEXED
add_executable(use_ply_data_DrawIndexed
use_ply_data_DrawIndexed.c
)
target_link_libraries(use_ply_data_DrawIndexed PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file_index)

#--------------MULTIPLE_INSTANCES_MULTIPLE_MESH_ONE_DRAW
add_executable(multiple_instances_multiple_mesh_one_draw
multiple_instances_multiple_mesh_one_draw.c
)
target_link_libraries(multiple_instances_multiple_mesh_one_draw PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)

#--------------MULTIPLE_INSTANCES_ONE_DRAW_WITH_COMPUTE
add_executable(multiple_instances_one_draw_with_compute
multiple_instances_one_draw_with_compute.c
)
target_link_libraries(multiple_instances_one_draw_with_compute PRIVATE glfw wgpu_native create_surface framework cglm read_ply_file)
