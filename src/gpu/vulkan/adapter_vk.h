#pragma once

#include "gpu/gpu.h"
#include "gpu/adapter.h"

#include <memory>

namespace gpu
{

class AdapterVk final : public IAdapter
{
public:
	virtual void* Alloc(const size_t size, const AllocationMode mode) override final;

	virtual void Free(void* const ptr) override final;

	virtual bool ExecuteKernel(
		const uint32_t id,
		const uint32_t nx,
		const uint32_t ny,
		const uint32_t nz,
		const void* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo) override final;

	static AdapterVk* CreateVulkanAdapter()
	{
		std::unique_ptr<AdapterVk> pAdapter(new AdapterVk);
		return pAdapter->Init() ? pAdapter.release() : nullptr;
	}

private:
	bool Init();

private:
	AdapterVk() = default;
	AdapterVk(AdapterVk&& ) = delete;
	AdapterVk(const AdapterVk& ) = delete;
	AdapterVk& operator=(AdapterVk&& ) = delete;
	AdapterVk& operator=(const AdapterVk& ) = delete;

private:
	// TODO: Add Vulkan-specific handles (VkInstance, VkDevice, VkCommandQueue, etc.)
};

extern IAdapter* CreateVulkanAdapter();

} // End of namespace gpu
