cmake_minimum_required(VERSION 3.0...3.25)
project(
	WebGPU_Examples # name of the project, which will also be the name of the visual studio solution if you use it
	VERSION 0.1.0 # any version number
	LANGUAGES CXX C # programming languages used by the project
)
# add_compile_options(-fsanitize=address)
# set (CMAKE_EXE_LINKER_FLAGS "-fsanitize=address")

#--------------WGPU_NATIVE
# send error for missing wgpu directory
if(EXISTS ${CMAKE_SOURCE_DIR}/wgpu)
  message( STATUS "wpu exists")
else()
  message( SEND_ERROR "\nYou will need to have wgpu-native. Get it from here: \nhttps://github.com/gfx-rs/wgpu-native/releases\nThen extract it to the root source directory and rename it to 'wgpu'\n" )
endif()
# assumes wgpu_native is downloaded and extracted to the root source directory and renamed 'wgpu' 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
add_library(wgpu_native STATIC IMPORTED)
find_library(WGPU_LIBRARY_LOC libwgpu_native.a libwgpu_native.lib PATHS 
"${CMAKE_SOURCE_DIR}/wgpu")
set_property(TARGET wgpu_native PROPERTY IMPORTED_LOCATION ${WGPU_LIBRARY_LOC})
target_include_directories(wgpu_native INTERFACE ${CMAKE_SOURCE_DIR}/wgpu)

#--------------CREATE_SURFACE
add_library(create_surface
create_surface.c
)
# target_include_directories(create_surface PUBLIC ${CMAKE_SOURCE_DIR}/1_getting_started)
target_link_libraries(create_surface PRIVATE glfw wgpu_native)
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(create_surface PRIVATE
    WGPU_TARGET=WGPU_TARGET_WINDOWS
    )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_compile_definitions(create_surface PRIVATE
    WGPU_TARGET=WGPU_TARGET_MACOS
    )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(create_surface PRIVATE
    WGPU_TARGET=WGPU_TARGET_LINUX_X11
    )
else()
    target_compile_definitions(create_surface PRIVATE
    WGPU_TARGET=NO_WGPU_TARGET
    )
endif()
#--------------FRAMEWORK
add_library(framework
framework.c
)
target_link_libraries(framework PRIVATE wgpu_native)
target_include_directories(framework PUBLIC ${CMAKE_SOURCE_DIR})

include(FetchContent)

#--------------RPLY
add_library(rply
rply.c
)
target_include_directories(rply PUBLIC ${CMAKE_SOURCE_DIR})

#--------------GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG        73948e6c0f15b1053cf74b7c4e6b04fd36e97e29
)
FetchContent_MakeAvailable(glfw)

#--------------CGLM
FetchContent_Declare(
        cglm
        GIT_REPOSITORY https://github.com/recp/cglm
)
FetchContent_MakeAvailable(cglm)

add_subdirectory(glm_cameras)
add_subdirectory(load_geometry_data)
add_subdirectory(shadertoy)
