project(
	Fundamentals # name of the project, which will also be the name of the visual studio solution if you use it
	VERSION 0.1.0 # any version number
	LANGUAGES CXX C # programming languages used by the project
)
add_compile_definitions(RESOURCE_DIR="${CMAKE_SOURCE_DIR}/shadertoy/res/")


#--------------SHADER_ART_CODING_INTRODUCTION
add_executable(shader_art_coding_introduction
shader_art_coding_introduction.c
)
target_link_libraries(shader_art_coding_introduction PRIVATE glfw wgpu_native create_surface framework)
