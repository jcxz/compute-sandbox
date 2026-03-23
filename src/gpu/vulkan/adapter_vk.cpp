#include "gpu/vulkan/adapter_vk.h"
#include <iostream>



namespace gpu
{

AdapterVk::~AdapterVk()
{
	mAllocations.clear();

	if (mDescriptorPool)
		vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);

	if (mCommandPool)
		vkDestroyCommandPool(mDevice, mCommandPool, nullptr);

	if (mDevice)
		vkDestroyDevice(mDevice, nullptr);

	if (mInstance)
		vkDestroyInstance(mInstance, nullptr);
}

void* AdapterVk::Alloc(const size_t size, const AllocationMode mode)
{
	// Initialize GPU if needed
	if (!Init())
	{
		std::cerr << "Failed to initialize GPU" << std::endl;
		return nullptr;
	}

	Allocation alloc;

	// Create buffer
	VkBufferCreateInfo bufferInfo;
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.flags = 0;
	bufferInfo.size = size;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferInfo.queueFamilyIndexCount = 0;
	bufferInfo.pQueueFamilyIndices = nullptr;

	if (vkCreateBuffer(mDevice, &bufferInfo, nullptr, &alloc.buffer) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Alloc() - Failed to create buffer!" << std::endl;
		return nullptr;
	}

	// Allocate memory
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(mDevice, alloc.buffer, &memRequirements);

	const VkMemoryPropertyFlags properties =
		(mode == AllocationMode::Device) ?
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT :
		(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	VkMemoryAllocateInfo allocInfo;
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.pNext = nullptr;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
	if (allocInfo.memoryTypeIndex == -1)
	{
		std::cerr << "AdapterVk::Alloc() - Failed to find suitable memory type!" << std::endl;
		vkDestroyBuffer(mDevice, alloc.buffer, nullptr);
		return nullptr;
	}

	if (vkAllocateMemory(mDevice, &allocInfo, nullptr, &alloc.memory) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Alloc() - Failed to allocate buffer memory!" << std::endl;
		vkDestroyBuffer(mDevice, alloc.buffer, nullptr);
		return nullptr;
	}

	if (vkBindBufferMemory(mDevice, alloc.buffer, alloc.memory, 0) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Alloc() - Failed to bind buffer memory!" << std::endl;
		vkDestroyBuffer(mDevice, alloc.buffer, nullptr);
		vkFreeMemory(mDevice, alloc.memory, nullptr);
		return nullptr;
	}

	// Map memory
	if (mode == AllocationMode::Shared)
	{
		if (vkMapMemory(mDevice, alloc.memory, 0, size, 0, &alloc.ptr) != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::Alloc() - Failed to map buffer memory!" << std::endl;
			vkDestroyBuffer(mDevice, alloc.buffer, nullptr);
			vkFreeMemory(mDevice, alloc.memory, nullptr);
			return nullptr;
		}
	}

	// Treat the buffer handle as the pointer reference for purely device local memory
	// if we aren't mapping it explicitly. For a shared abstraction, we need to return
	// something unique.
	void* ptr = (mode == AllocationMode::Shared) ? alloc.ptr : reinterpret_cast<void*>(alloc.buffer);

	mAllocations[ptr] = alloc;
	return ptr;
}

void AdapterVk::Free(void* const ptr)
{
	mAllocations.erase(ptr);
}

bool AdapterVk::ExecuteKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const uint32_t nz,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	// 1. Verify kernel ID
	if (id >= mKernels.size())
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Invalid kernel ID: " << id << std::endl;
		return false;
	}

	// 2. Setup the argument bindings into descriptor sets using pArgs and pArgsInfo
	// TODO: Create descriptor sets based on the reflection info of the arguments.

	// 3. Allocate a command buffer
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = mCommandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	if (vkAllocateCommandBuffers(mDevice, &allocInfo, &commandBuffer) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Failed to allocate command buffers!" << std::endl;
		return false;
	}

	// 4. Begin recording
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	// 5. Bind Pipeline and Descriptor sets
	const KernelInfo& kInfo = mKernels[id];
	if (kInfo.pipeline != VK_NULL_HANDLE)
	{
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kInfo.pipeline);
		// vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kInfo.pipelineLayout, ...);

		// 6. Dispatch
		vkCmdDispatch(commandBuffer, nx, ny, nz);
	}
	else
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Pipeline not generated (TODO: SPIRV loading)" << std::endl;
	}

	vkEndCommandBuffer(commandBuffer);

	// 7. Submit
	VkSubmitInfo submitInfo;
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = nullptr;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.pWaitSemaphores = nullptr;
	submitInfo.pWaitDstStageMask = nullptr;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.pSignalSemaphores = nullptr;

	vkQueueSubmit(mComputeQueue, 1, &submitInfo, VK_NULL_HANDLE);

	// 8. Wait (Synchronous for now)
	vkQueueWaitIdle(mComputeQueue);

	// 9. Cleanup
	vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);

	return true;
}

uint32_t AdapterVk::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
	return -1;
}

bool AdapterVk::IsInitialized() const
{
	// the command pool is initialized as last in the initialization process,
	// so if we have a valid command pool, we know the initialization has already succeeded in the past
	return mCommandPool != VK_NULL_HANDLE;
}

bool AdapterVk::Init()
{
	// 0. Check if already initialized
	if (IsInitialized())
		return true;

	// 1. Create Instance
	VkApplicationInfo appInfo;
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pNext = nullptr;
	appInfo.pApplicationName = "";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pNext = nullptr;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledLayerCount = 0;
	createInfo.ppEnabledLayerNames = nullptr;
	createInfo.enabledExtensionCount = 0;
	createInfo.ppEnabledExtensionNames = nullptr;

	if (vkCreateInstance(&createInfo, nullptr, &mInstance) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create Vulkan instance!" << std::endl;
		return false;
	}

	// 2. Pick Physical Device
	uint32_t deviceCount = 0;
	if (vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to find GPUs with Vulkan support!" << std::endl;
		return false;
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	if (vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data()) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to find GPUs with Vulkan support!" << std::endl;
		return false;
	}

	// Simple heuristic: just pick the first one, could be enhanced to prefer discrete GPUs
	mPhysicalDevice = devices[0];

	// 3. Find Compute Queue Family
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, queueFamilies.data());

	uint32_t computeQueueFamilyIndex = -1;
	for (uint32_t i = 0; i < queueFamilyCount; ++i)
	{
		if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			computeQueueFamilyIndex = i;
			break;
		}
	}

	if (computeQueueFamilyIndex == -1)
	{
		std::cerr << "AdapterVk::Init() - Failed to find a compute queue family!" << std::endl;
		return false;
	}

	// 4. Create Logical Device
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo;
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.pNext = nullptr;
	queueCreateInfo.flags = 0;
	queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkDeviceCreateInfo deviceCreateInfo;
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext = nullptr;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.enabledLayerCount = 0;
	deviceCreateInfo.ppEnabledLayerNames = nullptr;
	deviceCreateInfo.enabledExtensionCount = 0;
	deviceCreateInfo.ppEnabledExtensionNames = nullptr;
	deviceCreateInfo.pEnabledFeatures = nullptr;

	if (vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create logical device!" << std::endl;
		return false;
	}

	vkGetDeviceQueue(mDevice, computeQueueFamilyIndex, 0, &mComputeQueue);

	// 5. Create Command Pool
	VkCommandPoolCreateInfo poolInfo;
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.pNext = nullptr;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = computeQueueFamilyIndex;

	if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create command pool!" << std::endl;
		return false;
	}

	return true;
}

extern IAdapter* CreateVulkanAdapter()
{
	return AdapterVk::CreateVulkanAdapter();
}

} // End of namespace gpu
