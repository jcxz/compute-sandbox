#pragma once

#include "gpu/gpu.h"
#include "gpu/adapter.h"
#include "gpu/vulkan/vk.h"
#include <slang.h>
#include <slang-com-ptr.h>

#include <memory>
#include <unordered_map>
#include <vector>



namespace gpu
{

class AdapterVk final : public IAdapter
{
	struct KernelInfo
	{
		// TODO: Fields for VkPipeline, VkPipelineLayout, VkDescriptorSetLayout, etc.
		AdapterVk* pAdapter = nullptr;
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

		~KernelInfo()
		{
			vkDestroyPipeline(pAdapter->mDevice, pipeline, nullptr);
			vkDestroyPipelineLayout(pAdapter->mDevice, pipelineLayout, nullptr);
			for (VkDescriptorSetLayout layout : descriptorSetLayouts)
				vkDestroyDescriptorSetLayout(pAdapter->mDevice, layout, nullptr);
		}
	};

	struct Allocation
	{
		AdapterVk* pAdapter = nullptr;
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		void* ptr = nullptr;

		~Allocation()
		{
			if (ptr)
				vkUnmapMemory(pAdapter->mDevice, memory);
			vkDestroyBuffer(pAdapter->mDevice, buffer, nullptr);
			vkFreeMemory(pAdapter->mDevice, memory, nullptr);
		}
	};

public:
	virtual ~AdapterVk();

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
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	bool IsInitialized() const;
	bool Init();
	const KernelInfo* RequestKernel(const uint32_t id);
	bool BuildSlangProgram(const std::string& kernelName, slang::IComponentType** ppProgram);
	bool CreateDescriptorSetLayouts(slang::IComponentType* pProgram, std::vector<VkDescriptorSetLayout>& descriptorSetLayouts);

private:
	AdapterVk() = default;
	AdapterVk(AdapterVk&& ) = delete;
	AdapterVk(const AdapterVk& ) = delete;
	AdapterVk& operator=(AdapterVk&& ) = delete;
	AdapterVk& operator=(const AdapterVk& ) = delete;

private:
	//! Slang global session (basically a compiler instance)
	Slang::ComPtr<slang::IGlobalSession> mSlangGlobalSession;
	//! Slang session (keeps all compiler configuration)
	Slang::ComPtr<slang::ISession> mSlangSession;
	//! Vulkan instance
	VkInstance mInstance = VK_NULL_HANDLE;
	//! Vulkan physical device
	VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
	//! Vulkan logical device through which all GPU interaction happens
	VkDevice mDevice = VK_NULL_HANDLE;
	//! command queue for submittintg compute operations
	VkQueue mComputeQueue = VK_NULL_HANDLE;
	//! command pool for creating command buffers
	VkCommandPool mCommandPool = VK_NULL_HANDLE;
	//! Descriptor pool for allocating descriptor sets per kernel execution
	VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
	//! cache of precompiled pipelines ready to be used for starting a kernel
	std::vector<KernelInfo> mKernels;
	//! We treat the mapped pointer (or a dummy pointer if not mapped) as the allocation ID returned to the user
	std::unordered_map<void*, Allocation> mAllocations;
};

extern IAdapter* CreateVulkanAdapter();

} // End of namespace gpu
