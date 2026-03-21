#include "gpu/gpu.h"
#include "gpu/kernel_registry.h"
#include "gpu/adapter.h"


static gpu::IAdapter* gAdapter = nullptr;

namespace gpu
{

extern IAdapter* CreateMetalAdapter();

}

bool InitializeGpu(const GpuAdapterType type)
{
	if (gAdapter)
		return true;

	switch (type)
	{
#ifdef BUILD_METAL_ADAPTER
		case GpuAdapterType::Metal: gAdapter = gpu::CreateMetalAdapter(); break;
#endif
		default:
			break;
	}

	return gAdapter != nullptr;
}

void TerminateGpu()
{
	delete gAdapter;
	gAdapter = nullptr;
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
