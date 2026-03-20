cmake_minimum_required(VERSION 3.30)

set(PROJECT_NAME metal-test)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

set(HEADERS
	src/core/global.h
	src/core/gpu.h
	src/core/kernel.h
	src/core/kernel_cpu.h
	src/core/kernel_metal.h
	src/core/kernel_slang.h
	src/core/meta.h
	src/core/mtl.h
	src/core/preprocessor.h
	src/core/reflection.h
	src/kernels/add.h
	src/kernels/invert.h
	src/kernels/invert2.h
	src/kernels/invert3.h
	src/tests/common.h
)

set(SOURCES
	src/core/gpu.cpp
	src/core/mtl.cpp
	src/core/reflection.cpp
	src/tests/test1.cpp
	src/tests/test2.cpp
	src/tests/test3.cpp
	src/tests/test4.cpp
	src/tests/test5.cpp
	src/main.cpp
)

set(METAL_SHADERS
	src/kernels/add.metal
	src/kernels/invert.metal
	src/kernels/invert2.metal
	src/kernels/invert3.metal
)

set(SLANG_SHADERS
	src/kernels/invert3.slang
)

set(GENERATED_METAL_SHADERS)

foreach(SLANG ${SLANG_SHADERS})
	add_custom_command(
		OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SLANG}.metal"
		COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/slangc"
			"${CMAKE_CURRENT_SOURCE_DIR}/${SLANG}"
			-o "${CMAKE_BINARY_DIR}/shaders/${SLANG}.metal"
			-profile glsl_460
			-target metal
			-depfile "${CMAKE_BINARY_DIR}/shaders/${SLANG}.d"
			-I "${CMAKE_CURRENT_SOURCE_DIR}/src"
			-D__SLANG__=1
		DEPENDS ${SLANG}
		DEPFILE "${CMAKE_BINARY_DIR}/shaders/${SLANG}.d" # to make sure that the shader recompiles when only its includes change
		WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/slang-cache"
		COMMENT "Slang → Metal: ${CMAKE_BINARY_DIR}/shaders/${SLANG}.metal")
	list(APPEND GENERATED_METAL_SHADERS "${CMAKE_BINARY_DIR}/shaders/${SLANG}.metal")
endforeach()

set_source_files_properties(
	${METAL_SHADERS} ${GENERATED_METAL_SHADERS} PROPERTIES
	LANGUAGE METAL
	COMPILE_OPTIONS "-Isrc")

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${METAL_SHADERS} ${GENERATED_METAL_SHADERS} ${SLANG_SHADERS})

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage(
	NAME     metal-cpp
	URL      http://github.com/bkaradzic/metal-cpp/archive/refs/heads/metal-cpp_26.tar.gz
	URL_HASH SHA256=b0e6d0a39618f38049e915238ea1400aaea5292167385acc52a1666fd0509aab
)

target_include_directories(${PROJECT_NAME} PRIVATE
	${metal-cpp_SOURCE_DIR}
	src
)

target_link_libraries(${PROJECT_NAME}
	"-framework Foundation"
	"-framework QuartzCore"
	"-framework Metal"
)
