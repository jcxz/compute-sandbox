#include "gpu/vulkan/adapter_vk.h"
#include <iostream>



namespace gpu
{

IAdapter* CreateVulkanAdapter()
{
	return AdapterVk::CreateVulkanAdapter();
}

bool AdapterVk::Init()
{
	// TODO: Initialize Vulkan instance, device, queue
	std::cerr << "AdapterVk::Init() not implemented yet." << std::endl;
	return false;
}

void* AdapterVk::Alloc(const size_t size, const AllocationMode mode)
{
	// TODO: Implement Vulkan memory allocation
	std::cerr << "AdapterVk::Alloc() not implemented yet." << std::endl;
	return nullptr;
}

void AdapterVk::Free(void* const ptr)
{
	// TODO: Implement Vulkan memory free
	std::cerr << "AdapterVk::Free() not implemented yet." << std::endl;
}

bool AdapterVk::ExecuteKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const uint32_t nz,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	// TODO: Implement Vulkan kernel execution
	std::cerr << "AdapterVk::ExecuteKernel() not implemented yet." << std::endl;
	return false;
}

} // End of namespace gpu
