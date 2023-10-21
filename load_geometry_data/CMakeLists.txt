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