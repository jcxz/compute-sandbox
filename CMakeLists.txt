cmake_minimum_required(VERSION 3.30)

set(PROJECT_NAME compute-sandbox)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

# --- CONFIGURATION OPTIONS ---
if(APPLE)
	option(BUILD_METAL_ADAPTER "Build Metal adapter" ON)
	option(BUILD_VULKAN_ADAPTER "Build Vulkan adapter" OFF)
else()
	option(BUILD_METAL_ADAPTER "Build Metal adapter" OFF)
	option(BUILD_VULKAN_ADAPTER "Build Vulkan adapter" ON)
endif()

# --- COMMON EXECUTABLE ---
set(HEADERS
	src/core/global.h
	src/core/meta.h
	src/core/preprocessor.h
	src/core/reflection.h
	src/gpu/gpu.h
	src/gpu/adapter.h
	src/gpu/kernel.h
	src/gpu/kernel_cpu.h
	src/gpu/kernel_metal.h
	src/gpu/kernel_registry.h
	src/gpu/kernel_slang.h
	src/tests/invert/invert.h
)

set(SOURCES
	src/core/reflection.cpp
	src/gpu/gpu.cpp
	src/tests/invert/test.cpp
	src/main.cpp
)

set(SLANG_SHADERS
	src/tests/invert/invert.slang
)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${SLANG_SHADERS})
target_include_directories(${PROJECT_NAME} PRIVATE src)

# --- METAL ADAPTER ---
if (BUILD_METAL_ADAPTER)
	set(HEADERS_METAL
		src/gpu/metal/adapter_mtl.h
		src/gpu/metal/mtl.h
	)

	set(SOURCES_METAL
		src/gpu/metal/adapter_mtl.cpp
		src/gpu/metal/mtl.cpp
	)

	set(METAL_SHADERS
		src/tests/invert/invert.metal
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
			WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
			COMMENT "Slang → Metal: ${CMAKE_BINARY_DIR}/shaders/${SLANG}.metal")
		list(APPEND GENERATED_METAL_SHADERS "${CMAKE_BINARY_DIR}/shaders/${SLANG}.metal")
	endforeach()

	set_source_files_properties(
		${METAL_SHADERS} ${GENERATED_METAL_SHADERS} PROPERTIES
		LANGUAGE METAL
		COMPILE_OPTIONS "-Isrc")

	target_sources(${PROJECT_NAME} PRIVATE
		${HEADERS_METAL}
		${SOURCES_METAL}
		${METAL_SHADERS}
		${GENERATED_METAL_SHADERS}
	)

	# add dependencies
	include(cmake/CPM.cmake)

	CPMAddPackage(
		NAME     metal-cpp
		URL      http://github.com/bkaradzic/metal-cpp/archive/refs/heads/metal-cpp_26.tar.gz
		URL_HASH SHA256=b0e6d0a39618f38049e915238ea1400aaea5292167385acc52a1666fd0509aab
	)

	target_include_directories(${PROJECT_NAME} PRIVATE ${metal-cpp_SOURCE_DIR})

	target_compile_definitions(${PROJECT_NAME} PRIVATE "BUILD_METAL_ADAPTER")

	target_link_libraries(${PROJECT_NAME} PRIVATE
		"-framework Foundation"
		"-framework QuartzCore"
		"-framework Metal"
	)
endif()

# --- VULKAN ADAPTER ---
if (BUILD_VULKAN_ADAPTER)
	set(HEADERS_VULKAN
		src/gpu/vulkan/adapter_vk.h
		src/gpu/vulkan/vk.h
	)

	set(SOURCES_VULKAN
		src/gpu/vulkan/adapter_vk.cpp
	)

	target_sources(${PROJECT_NAME} PRIVATE
		${HEADERS_VULKAN}
		${SOURCES_VULKAN}
	)

	target_compile_definitions(${PROJECT_NAME} PRIVATE "BUILD_VULKAN_ADAPTER")

	find_package(Vulkan)
	if(Vulkan_FOUND)
		target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
	else()
		message(FATAL_ERROR "Vulkan not found! BUILD_VULKAN_ADAPTER is enabled but Vulkan SDK is missing.")
	endif()
endif()