# webgpu examples c
> **Note:** I'm freezing the wgpu version at `v0.17.0.2` because the [webgpu native API](https://github.com/webgpu-native/webgpu-headers) is still very much in flux and it's alot of work to update all of the examples for breaking changes.


This project is more of an extension on the [webgpu_fundamentals_c](https://github.com/bobajeff/webgpu_fundamentals_c). The examples here aren't from https://webgpufundamentals.org/ and are more exersizes I've been trying or in the case of the shadertoy example just something I've ported. Still just learning wgpu.

## building
### get the source
```
git clone git@github.com:bobajeff/webgpu_examples_c.git
```
### get the dependecies
* wgpu-native v0.17.0.2 (required) -
    * Download the [wgpu-native v0.17.0.2](https://github.com/gfx-rs/wgpu-native/releases/tag/v0.17.0.2) build
    * extract in source directory 
    * rename folder to `wgpu`

### build the project
In source directory run:
```
cmake -B build
cmake --build build
```

### running examples
Run the executables found in their respective build subdirectories or run the `test_build.sh` script