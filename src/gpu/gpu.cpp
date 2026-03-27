#include "gpu/gpu.h"
#include "gpu/kernel_registry.h"
#include "gpu/adapter.h"
#ifdef _WIN32
#include <Windows.h>
#endif
// RenderDoc currently does not support MacOS
#ifndef __APPLE__
#include "RenderDoc/renderdoc_app.h"
#endif

#include <iostream>



#ifndef __APPLE__
// for more see: https://renderdoc.org/docs/in_application_api.html
static RENDERDOC_API_1_1_2* gRdocApi = nullptr;
#endif
// a pointer to the GPU adapter 	instance
static gpu::IAdapter* gAdapter = nullptr;
// a flag indicating whether the GPU api was initialized successfully
static bool gIsGpuInitialized = false;

namespace gpu
{

extern IAdapter* CreateMetalAdapter(const bool debugMode);
extern IAdapter* CreateVulkanAdapter(const bool debugMode);

}

#ifndef __APPLE__
static bool InitRDocCapture()
{
#ifdef _WIN32
	if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
#else
	// At init, on linux/android.
	// For android replace librenderdoc.so with libVkLayer_GLES_RenderDoc.so
	if(void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
#endif
	{
		const pRENDERDOC_GetAPI RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(reinterpret_cast<void*>(GetProcAddress(mod, "RENDERDOC_GetAPI")));
		const int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&gRdocApi);
		std::cerr << "RENDERDOC_GetAPI returned " << ret << std::endl;
		return ret == 1;
	}

	// RenderDoc is not present (most likely we are not running via RenderDoc)
	return true;
}
#endif

static bool InitGpuAdapter(const uint32_t flags, const GpuAdapterType type)
{
	switch (type)
	{
		case GpuAdapterType::Default:
			// pick the first enabled Adapter whichever that may be
#ifdef BUILD_METAL_ADAPTER
		case GpuAdapterType::Metal: gAdapter = gpu::CreateMetalAdapter(flags & GpuFlags::Debug); break;
#endif
#ifdef BUILD_VULKAN_ADAPTER
		case GpuAdapterType::Vulkan: gAdapter = gpu::CreateVulkanAdapter(flags & GpuFlags::Debug); break;
#endif
		default:
			break;
	}

	return gAdapter != nullptr;
}

bool InitializeGpu(const uint32_t flags, const GpuAdapterType type)
{
	if (gIsGpuInitialized)
		return true;

	if (flags & GpuFlags::RDocCapture)
	{
#ifndef __APPLE__
		if (!InitRDocCapture())
			return false;
#endif
	}

	if (!InitGpuAdapter(flags, type))
		return false;

	gIsGpuInitialized = true;
	return true;
}

void TerminateGpu()
{
	delete gAdapter;
	gAdapter = nullptr;
}

void BeginGpuCapture()
{
#ifndef __APPLE__
	if (gRdocApi)
		gRdocApi->StartFrameCapture(nullptr, nullptr);
#endif
}

void EndGpuCapture()
{
#ifndef __APPLE__
	if (gRdocApi)
		gRdocApi->EndFrameCapture(nullptr, nullptr);
#endif
}

void* GpuAlloc(const size_t size, const AllocationMode mode)
{
	return gAdapter ? gAdapter->Alloc(size, mode) : nullptr;
}

void GpuFree(void* const ptr)
{
	if (gAdapter)
		gAdapter->Free(ptr);
}

uint32_t RegisterKernel(const std::string& name)
{
	return KernelRegistry::GetInstance()->New(name);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return gAdapter && gAdapter->ExecuteKernel(id, nx, 1, 1, pArgs, pArgsInfo);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return gAdapter && gAdapter->ExecuteKernel(id, nx, ny, 1, pArgs, pArgsInfo);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const uint32_t nz,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return gAdapter && gAdapter->ExecuteKernel(id, nx, ny, nz, pArgs, pArgsInfo);
}
