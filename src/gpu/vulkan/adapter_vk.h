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
	//! stores all the information about a single allocation made via Alloc()
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

		explicit Allocation(AdapterVk* pAdapter = nullptr)
			: pAdapter(pAdapter)
		{ }

		Allocation(const Allocation& ) = delete;
		Allocation& operator=(const Allocation& ) = delete;

		Allocation(Allocation&& other) noexcept
			: pAdapter(other.pAdapter)
			, buffer(other.buffer)
			, memory(other.memory)
			, deviceAddress(other.deviceAddress)
			, ptr(other.ptr)
		{
			other.pAdapter = nullptr;
			other.buffer = VK_NULL_HANDLE;
			other.memory = VK_NULL_HANDLE;
			other.deviceAddress = 0;
			other.ptr = nullptr;
		}

		Allocation& operator=(Allocation&& other) noexcept
		{
			std::swap(pAdapter, other.pAdapter);
			std::swap(buffer, other.buffer);
			std::swap(memory, other.memory);
			std::swap(deviceAddress, other.deviceAddress);
			std::swap(ptr, other.ptr);
			return *this;
		}

		~Allocation()
		{
			if (pAdapter)
			{
				if (ptr)
					vkUnmapMemory(pAdapter->mDevice, memory);
				vkDestroyBuffer(pAdapter->mDevice, buffer, nullptr);
				vkFreeMemory(pAdapter->mDevice, memory, nullptr);
			}
		}
	};

	//! this structure contains reflection information about the kernel so that we know how to fill up the arguments buffer
	// ! and setup the dispatch call
	struct KernelReflectionInfo
	{
		//! thread group size, so that we know how to configure the dispatch command
		//! This can be made configurable from shader
		SlangUInt threadGroupSize[3];
		//! size of the arguments buffer
		size_t argsBufferSize = 0;
		//! Layout of the kernel arguments, so that we know how to fill them up
		std::vector<size_t> offsets;

		KernelReflectionInfo() = default;
		KernelReflectionInfo(const KernelReflectionInfo& ) = default;
		KernelReflectionInfo& operator=(const KernelReflectionInfo& ) = default;

		KernelReflectionInfo(KernelReflectionInfo&& other) noexcept
			: threadGroupSize{other.threadGroupSize[0], other.threadGroupSize[1], other.threadGroupSize[2]}
			, argsBufferSize(other.argsBufferSize)
			, offsets(std::move(other.offsets))
		{ }

		KernelReflectionInfo& operator=(KernelReflectionInfo&& other) noexcept
		{
			threadGroupSize[0] = other.threadGroupSize[0];
			threadGroupSize[1] = other.threadGroupSize[1];
			threadGroupSize[2] = other.threadGroupSize[2];
			argsBufferSize = other.argsBufferSize;
			offsets = std::move(other.offsets);
			return *this;
		}
	};

	//! this structure contains cached information about the kernel
	struct KernelInfo
	{
		//! a back pointer to the adapter to be able to access mDevice
		AdapterVk* pAdapter = nullptr;
		//! GPU allocated memory for the kernel arguments
		void* pArgsBuffer = nullptr;
		//! Vulkan device address of the kernel arguments buffer
		VkDeviceAddress argsBufferDeviceAddress = 0;
		//! Vulkan handle to the compute pipeline created for this kernel
		VkPipeline pipeline = VK_NULL_HANDLE;
		//! Vulkan handle to the pipeline layout created for this kernel
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		//! Reflection information about the kernel
		KernelReflectionInfo reflectionInfo;

		explicit KernelInfo(AdapterVk* pAdapter = nullptr)
			: pAdapter(pAdapter)
		{ }

		KernelInfo(const KernelInfo& ) = delete;
		KernelInfo& operator=(const KernelInfo& ) = delete;

		KernelInfo(KernelInfo&& other) noexcept
			: pAdapter(other.pAdapter)
			, pArgsBuffer(other.pArgsBuffer)
			, argsBufferDeviceAddress(other.argsBufferDeviceAddress)
			, pipeline(other.pipeline)
			, pipelineLayout(other.pipelineLayout)
			, reflectionInfo(std::move(other.reflectionInfo))
		{
			other.pAdapter = nullptr;
			other.pArgsBuffer = nullptr;
			other.argsBufferDeviceAddress = 0;
			other.pipeline = VK_NULL_HANDLE;
			other.pipelineLayout = VK_NULL_HANDLE;
		}

		KernelInfo& operator=(KernelInfo&& other) noexcept
		{
			std::swap(pAdapter, other.pAdapter);
			std::swap(pArgsBuffer, other.pArgsBuffer);
			std::swap(argsBufferDeviceAddress, other.argsBufferDeviceAddress);
			std::swap(pipeline, other.pipeline);
			std::swap(pipelineLayout, other.pipelineLayout);
			reflectionInfo = std::move(other.reflectionInfo);
			return *this;
		}

		~KernelInfo()
		{
			if (pAdapter)
			{
				pAdapter->Free(pArgsBuffer);
				vkDestroyPipeline(pAdapter->mDevice, pipeline, nullptr);
				vkDestroyPipelineLayout(pAdapter->mDevice, pipelineLayout, nullptr);
			}
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

	static AdapterVk* CreateVulkanAdapter(const bool debugMode = false)
	{
		std::unique_ptr<AdapterVk> pAdapter(new AdapterVk);
		return pAdapter->Init(debugMode) ? pAdapter.release() : nullptr;
	}

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	bool IsInitialized() const;
	bool Init(const bool debugMode);
	const KernelInfo* RequestKernel(const uint32_t id);
	//! compiles shader program
	bool BuildSlangProgram(const std::string& kernelName, slang::IComponentType** ppProgram, VkShaderModule& pShaderModule);
	//! validates shader and builds reflection info
	bool ReflectSlangProgram(slang::IComponentType* pProgram, KernelReflectionInfo& reflectionInfo);
	//! fills the arguments buffer with the arguments
	bool EncodeKernelArguments(const KernelInfo* const pKernel, const uint8_t* const pArgs, const refl::TypeMetaInfo* const pArgsInfo);
	//! returns the buffer associated with the given pointer
	const Allocation* GetAllocation(void* const ptr) const;

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
	//! pipeline cache to speed up creating compute pipelines for kernels
	VkPipelineCache mPipelineCache = VK_NULL_HANDLE;
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
};

extern IAdapter* CreateVulkanAdapter(const bool debugMode = false);

} // End of namespace gpu
