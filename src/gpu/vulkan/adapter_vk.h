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
	static_assert(sizeof(VkDeviceAddress) == sizeof(uint64_t), "VkDeviceAddress not the same size as uint64_t");
	static_assert(sizeof(void*) == sizeof(uint64_t), "void* not the same size as uint64_t");

private:
	//! this structure contains reflection information about the kernel so that we know how to fill up the arguments buffer
	// ! and setup the dispatch call
	struct KernelReflectionInfo
	{
		//! thread group size, so that we know how to configure the dispatch command
		//! This can be made configurable from shader
		SlangUInt threadGroupSize[3];
		//! Layout of the kernel arguments, so that we know how to fill them up
		std::vector<size_t> offsets;

		void Swap(KernelReflectionInfo& other)
		{
			std::swap(threadGroupSize, other.threadGroupSize);
			std::swap(offsets, other.offsets);
		}
	};

	//! this structure contains cached information about the kernel
	struct KernelInfo
	{
		AdapterVk* pAdapter = nullptr;
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		// TODO: descriptor set layout ani descriptor sety nepotrebujem,
		// lebo ocakavam len jeden custom argument -> ktorym je pointer na Args buffer,
		// ktory mi slang prelozi ako push constants
		// Ale pre vseobecny pripad pripad, ked by som chcel mat aj textury by som mozno
		// mohol nechal descriptor set layouty
		// zaroven budem musiet pridat pointer na reflexiu, lebo budem musiet
		// nieco ako EncodeKernelArguments v AdapterMtl, cize budem musiet v runtime zakazdym preliezt
		// cez vsetky vstupne argumenty a zistit kam, na ake offsety ich mam napchat
		// este by som tu mohol pridat aj threadGroupSize z kernelu, aby som vedel ako mam nasetupovat dispatch
		std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
		//! cached descriptor sets (in our case we will always have only one descriptor set with one binding,
		//! because at the moment we only allow kernels with one argument that is a structure of scalars and pointers)
		std::vector<VkDescriptorSet> descriptorSets;
		//! buffer for scalar kernel arguments
		VkBuffer buffer = VK_NULL_HANDLE;
		//! Vulkan memory backing the buffer
		VkDeviceMemory memory = VK_NULL_HANDLE;
		//! the device address of the buffer
		VkDeviceAddress deviceAddress = 0;
		//! CPU mapped pointer for the buffer
		void* ptr = nullptr;
		//! Reflection information about the kernel
		KernelReflectionInfo reflectionInfo;

		//KernelInfo() = default;
		//KernelInfo(KernelInfo&& other);
		//KernelInfo& operator=(KernelInfo&& other);

		void Swap(KernelInfo& other)
		{
			std::swap(pAdapter, other.pAdapter);
			std::swap(pipeline, other.pipeline);
			std::swap(pipelineLayout, other.pipelineLayout);
			std::swap(descriptorSetLayouts, other.descriptorSetLayouts);
			std::swap(descriptorSets, other.descriptorSets);
			std::swap(buffer, other.buffer);
			std::swap(memory, other.memory);
			std::swap(ptr, other.ptr);
			std::swap(reflectionInfo, other.reflectionInfo);
		}

		~KernelInfo()
		{
			if (pAdapter)
			{
				// free the buffer resource
				if (ptr)
					vkUnmapMemory(pAdapter->mDevice, memory);
				vkDestroyBuffer(pAdapter->mDevice, buffer, nullptr);
				vkFreeMemory(pAdapter->mDevice, memory, nullptr);
				// free descriptor sets
				//vkFreeDescriptorSets(pAdapter->mDevice, pAdapter->mDescriptorPool, descriptorSets.size(), descriptorSets.data());
				// free the all the layout descriptors
				vkDestroyPipeline(pAdapter->mDevice, pipeline, nullptr);
				vkDestroyPipelineLayout(pAdapter->mDevice, pipelineLayout, nullptr);
				for (VkDescriptorSetLayout layout : descriptorSetLayouts)
					vkDestroyDescriptorSetLayout(pAdapter->mDevice, layout, nullptr);
			}
		}
	};

	struct Allocation
	{
		//! a back pointer to the adapter
		AdapterVk* pAdapter = nullptr;
		//! Vulkan buffer backing this allocation
		VkBuffer buffer = VK_NULL_HANDLE;
		//! Vulkan memory backing this allocation
		VkDeviceMemory memory = VK_NULL_HANDLE;
		//! the device address of the buffer
		VkDeviceAddress deviceAddress = 0;
		//! CPU mapped pointer to the buffer
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

	static AdapterVk* CreateVulkanAdapter(const bool debugMode = true)
	{
		std::unique_ptr<AdapterVk> pAdapter(new AdapterVk);
		pAdapter->mDebugMode = debugMode;
		return pAdapter->Init() ? pAdapter.release() : nullptr;
	}

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	bool IsInitialized() const;
	bool Init();
	const KernelInfo* RequestKernel(const uint32_t id);
	bool BuildSlangProgram(const std::string& kernelName, slang::IComponentType** ppProgram);
	//! validates shader and builds reflection info
	bool ReflectSlangProgram(slang::IComponentType* pProgram, KernelReflectionInfo& reflectionInfo);
	VkShaderModule CreateShaderModule(slang::IComponentType* pProgram);
	VkPipelineLayout CreatePipelineLayout(slang::IComponentType* pProgram);
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
	//! Debug messenger for Vulkan validation layers
	VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
	//! cache of precompiled pipelines ready to be used for starting a kernel
	std::vector<KernelInfo> mKernels;
	//! We treat the mapped pointer (or a dummy pointer if not mapped) as the allocation ID returned to the user
	std::unordered_map<void*, Allocation> mAllocations;
	//! whether debugging is enabled
	bool mDebugMode = false;
};

extern IAdapter* CreateVulkanAdapter();

} // End of namespace gpu
