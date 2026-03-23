#include "gpu/vulkan/adapter_vk.h"
#include "gpu/kernel_registry.h"
#include "core/global.h"
#include <iostream>
#include <fstream>
#include <vector>



static VkDescriptorType MapSlangTypeToVulkanDescriptorType(slang::TypeReflection* type)
{
	const auto kind = type->getKind();
	switch (kind)
	{
		case slang::TypeReflection::Kind::Resource:
		{
			const auto resourceType = type->getResourceShape();
			const auto access = type->getResourceAccess();
			// Simplified mapping, can be expanded for textures, etc.
			return (access == SLANG_RESOURCE_ACCESS_READ_WRITE) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		}
		case slang::TypeReflection::Kind::ConstantBuffer:
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		default:
			return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}
}

namespace gpu
{

AdapterVk::~AdapterVk()
{
	mAllocations.clear();

	mKernels.clear();

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
	// Initialize GPU if needed
	if (!Init())
	{
		std::cerr << "Failed to initialize GPU" << std::endl;
		return false;
	}

	// 1. Get/Build Pipeline
	const KernelInfo* pKernel = RequestKernel(id);
	if (pKernel == nullptr)
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Failed to initialize GPU data for kernel "
				  << KernelRegistry::GetInstance()->GetKernelName(id) << " (id: " << id << ")" << std::endl;
		return false;
	}

	// 2. Allocate descriptor sets
	VkDescriptorSetAllocateInfo descriptorSetAllocInfo;
	descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetAllocInfo.pNext = nullptr;
	descriptorSetAllocInfo.descriptorPool = mDescriptorPool;
	descriptorSetAllocInfo.descriptorSetCount = pKernel->descriptorSetLayouts.size();
	descriptorSetAllocInfo.pSetLayouts = pKernel->descriptorSetLayouts.data();

	std::vector<VkDescriptorSet> descriptorSets(pKernel->descriptorSetLayouts.size());
	if (vkAllocateDescriptorSets(mDevice, &descriptorSetAllocInfo, descriptorSets.data()) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Failed to allocate descriptor sets!" << std::endl;
		return false;
	}

	// 3. Fill descriptor sets with kernel arguments

	// 4. Allocate a command buffer
	VkCommandBufferAllocateInfo commandBufferAllocInfo;
	commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocInfo.pNext = nullptr;
	commandBufferAllocInfo.commandPool = mCommandPool;
	commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	if (vkAllocateCommandBuffers(mDevice, &commandBufferAllocInfo, &commandBuffer) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Failed to allocate command buffers!" << std::endl;
		return false;
	}

	// 5. Begin recording
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	// 6. Bind Pipeline and Descriptor sets
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pKernel->pipeline);
	vkCmdBindDescriptorSets(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		pKernel->pipelineLayout,
		0,
		descriptorSets.size(),
		descriptorSets.data(),
		0,
		nullptr);

	// 7. Dispatch
	vkCmdDispatch(commandBuffer, nx, ny, nz);

	vkEndCommandBuffer(commandBuffer);

	// 8. Submit
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

	// 9. Wait (Synchronous for now)
	vkQueueWaitIdle(mComputeQueue);

	// 10. Cleanup
	vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
	vkResetDescriptorPool(mDevice, mDescriptorPool, 0);

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

const AdapterVk::KernelInfo* AdapterVk::RequestKernel(const uint32_t id)
{
	EM_ASSERT((mKernels.size() == KernelRegistry::GetInstance()->GetKernelCount()) && "Kernel registry size mismatch");

	const std::string& kernelName = KernelRegistry::GetInstance()->GetKernelName(id);
	if (kernelName.empty())
	{
		std::cerr << "Invalid kernel id " << id << std::endl;
		return nullptr;
	}

	auto& kernel = mKernels[id];
	if (kernel.pipeline == VK_NULL_HANDLE)
	{
		// build slang program
		Slang::ComPtr<slang::IComponentType> pProgram;
		if (!BuildSlangProgram(kernelName, pProgram.writeRef()))
		{
			std::cerr << "Failed to build Slang program for " << kernelName << std::endl;
			return nullptr;
		}

		Slang::ComPtr<slang::IBlob> pSPIRVCode;
		{
			Slang::ComPtr<slang::IBlob> pDiagnostics;
			pProgram->getEntryPointCode(0, 0, pSPIRVCode.writeRef(), pDiagnostics.writeRef());
			if (!pSPIRVCode)
			{
				std::cerr << "Failed to get SPIR-V code from Slang program." << std::endl;
				if (pDiagnostics)
					std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
				return nullptr;
			}
		}

		// create descriptor set layouts using slang's reflection API
		std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
		if (!CreateDescriptorSetLayouts(pProgram, descriptorSetLayouts))
		{
			std::cerr << "Failed to create descriptor set layouts for " << kernelName << std::endl;
			return nullptr;
		}

		// create pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutInfo;
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.pNext = nullptr;
		pipelineLayoutInfo.flags = 0;
		pipelineLayoutInfo.setLayoutCount = (uint32_t)descriptorSetLayouts.size();
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout pipelineLayout;
		if (vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			std::cerr << "Failed to create pipeline layout for " << kernelName << std::endl;
			for (auto& descriptorSetLayout : descriptorSetLayouts)
				vkDestroyDescriptorSetLayout(mDevice, descriptorSetLayout, nullptr);
			return nullptr;
		}

		VkShaderModuleCreateInfo moduleCreateInfo{};
		moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleCreateInfo.codeSize = pSPIRVCode->getBufferSize();
		moduleCreateInfo.pCode = (const uint32_t*)pSPIRVCode->getBufferPointer();

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(mDevice, &moduleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			std::cerr << "Failed to create shader module for " << kernelName << std::endl;
			vkDestroyPipelineLayout(mDevice, pipelineLayout, nullptr);
			for (auto& descriptorSetLayout : descriptorSetLayouts)
				vkDestroyDescriptorSetLayout(mDevice, descriptorSetLayout, nullptr);
			return nullptr;
		}

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = kernelName.c_str();

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.stage = shaderStageInfo;
		pipelineInfo.layout = pipelineLayout;

		VkPipeline pipeline;
		if (vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
		{
			std::cerr << "Failed to create compute pipeline for " << kernelName << std::endl;
			vkDestroyShaderModule(mDevice, shaderModule, nullptr);
			vkDestroyPipelineLayout(mDevice, pipelineLayout, nullptr);
			for (auto& descriptorSetLayout : descriptorSetLayouts)
				vkDestroyDescriptorSetLayout(mDevice, descriptorSetLayout, nullptr);
			return nullptr;
		}

		vkDestroyShaderModule(mDevice, shaderModule, nullptr);

		kernel.pAdapter = this;
		kernel.descriptorSetLayouts = std::move(descriptorSetLayouts);
		kernel.pipelineLayout = pipelineLayout;
		kernel.pipeline = pipeline;
	}

	return &kernel;
}

bool AdapterVk::BuildSlangProgram(const std::string& kernelName, slang::IComponentType** ppProgram)
{
	// compile slang shader
	Slang::ComPtr<slang::IModule> pModule;
	{
		Slang::ComPtr<slang::IBlob> pDiagnostics;
		pModule = mSlangSession->loadModule(kernelName.c_str(), pDiagnostics.writeRef());
		if (!pModule)
		{
			std::cerr << "Failed to load Slang module: " << kernelName << std::endl;
			if (pDiagnostics)
				std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
			return false;
		}
	}

	Slang::ComPtr<slang::IEntryPoint> pEntryPoint;
	if (pModule->findEntryPointByName(kernelName.c_str(), pEntryPoint.writeRef()) != SLANG_OK)
	{
		std::cerr << "Failed to find entry point: " << kernelName << std::endl;
		return false;
	}

	// link slang shader
	{
		slang::IComponentType* components[] = { pModule, pEntryPoint };
		Slang::ComPtr<slang::IBlob> pDiagnostics;
		if (mSlangSession->createCompositeComponentType(components, SLANG_COUNT_OF(components), ppProgram, pDiagnostics.writeRef()) != SLANG_OK)
		{
			std::cerr << "Failed to create Slang composite component program." << std::endl;
			if (pDiagnostics)
				std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
			return false;
		}
	}

	//Slang::ComPtr<slang::IComponentType> pLinkedProgram;
	//{
	//	Slang::ComPtr<slang::IBlob> pDiagnostics;
	//	pProgram->link(pLinkedProgram.writeRef(), pDiagnostics.writeRef());
	//	if (!pLinkedProgram)
	//	{
	//		std::cerr << "Failed to link Slang program." << std::endl;
	//		if (pDiagnostics)
	//			std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
	//		return nullptr;
	//	}
	//}

	return true;
}

bool AdapterVk::CreateDescriptorSetLayouts(slang::IComponentType* pProgram, std::vector<VkDescriptorSetLayout>& descriptorSetLayouts)
{
	std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> setBindings;

	slang::ProgramLayout* pReflection = pProgram->getLayout();
	slang::EntryPointReflection* pEntryPointReflection = pReflection->getEntryPointByIndex(0);
	const uint32_t parameterCount = pEntryPointReflection->getParameterCount();
	for (uint32_t i = 0; i < parameterCount; ++i)
	{
		slang::VariableLayoutReflection* pParam = pEntryPointReflection->getParameterByIndex(i);

		VkDescriptorSetLayoutBinding binding;
		binding.binding = (uint32_t) pParam->getBindingIndex();
		binding.descriptorType = MapSlangTypeToVulkanDescriptorType(pParam->getType());
		binding.descriptorCount = 1; // Simplified
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		binding.pImmutableSamplers = nullptr;

		setBindings[pParam->getBindingSpace()].push_back(binding);
	}

	uint32_t maxSetIndex = 0;
	for (auto const& [setIndex, bindings] : setBindings)
		maxSetIndex = std::max(maxSetIndex, setIndex);

	descriptorSetLayouts.resize(maxSetIndex + 1, VK_NULL_HANDLE);

	for (auto const& [setIndex, bindings] : setBindings)
	{
		VkDescriptorSetLayoutCreateInfo layoutInfo;
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.pNext = nullptr;
		layoutInfo.flags = 0;
		layoutInfo.bindingCount = (uint32_t)bindings.size();
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &descriptorSetLayouts[setIndex]) != VK_SUCCESS)
		{
			std::cerr << "Failed to create descriptor set layout for set " << setIndex << std::endl;
			return false;
		}
	}

	return true;
}

bool AdapterVk::IsInitialized() const
{
	// the descriptor pool is initialized as last in the initialization process,
	// so if we have a valid descriptor pool, we know the initialization has already succeeded in the past
	return mDescriptorPool != VK_NULL_HANDLE;
}

bool AdapterVk::Init()
{
	static const char* const kIncludePaths[] = {
		"src"
	};

	static const slang::PreprocessorMacroDesc kPreprocessorMacros[] =
	{
		{ "__SLANG__", "1" }
	};

	// 0. Check if already initialized
	if (IsInitialized())
		return true;

	// 1. Initialize Slang
	if (slang::createGlobalSession(mSlangGlobalSession.writeRef()) != SLANG_OK)
	{
		std::cerr << "AdapterVk::Init() - Failed to create Slang global session!" << std::endl;
		return false;
	}

	slang::TargetDesc targetDesc;
	targetDesc.format = SLANG_SPIRV;
	targetDesc.profile = mSlangGlobalSession->findProfile("glsl_460"); // spirv_1_5
	targetDesc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

	slang::SessionDesc sessionDesc;
	sessionDesc.targets = &targetDesc;
	sessionDesc.targetCount = 1;
	sessionDesc.searchPaths = kIncludePaths;
	sessionDesc.searchPathCount = SLANG_COUNT_OF(kIncludePaths);
	sessionDesc.preprocessorMacros = kPreprocessorMacros;
	sessionDesc.preprocessorMacroCount = SLANG_COUNT_OF(kPreprocessorMacros);

	if (mSlangGlobalSession->createSession(sessionDesc, mSlangSession.writeRef()) != SLANG_OK)
	{
		std::cerr << "AdapterVk::Init() - Failed to create Slang session!" << std::endl;
		return false;
	}

	// 2. Create Instance
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

	// 3. Pick Physical Device
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

	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(mPhysicalDevice, &deviceFeatures);

	// 4. Find Compute Queue Family
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

	// 5. Create Logical Device
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
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;   // lets enable all supported device features

	if (vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create logical device!" << std::endl;
		return false;
	}

	vkGetDeviceQueue(mDevice, computeQueueFamilyIndex, 0, &mComputeQueue);

	// 6. Create Command Pool
	VkCommandPoolCreateInfo commandPoolInfo;
	commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolInfo.pNext = nullptr;
	commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;

	if (vkCreateCommandPool(mDevice, &commandPoolInfo, nullptr, &mCommandPool) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create command pool!" << std::endl;
		return false;
	}

	// 7. Create Descriptor Pool
	const VkDescriptorPoolSize poolSizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
	};

	VkDescriptorPoolCreateInfo descriptorPoolInfo;
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.pNext = nullptr;
	descriptorPoolInfo.flags = 0;
	descriptorPoolInfo.maxSets = 128;
	descriptorPoolInfo.poolSizeCount = 2;
	descriptorPoolInfo.pPoolSizes = poolSizes;

	if (vkCreateDescriptorPool(mDevice, &descriptorPoolInfo, nullptr, &mDescriptorPool) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create descriptor pool!" << std::endl;
		return false;
	}

	// ensure the cache is large enough for all kernels
	// all CPU kernels have a static ID variable intialized via kernel registry
	// before main is started, so at this point all kernels
	// should be registered and this should be a valid code
	mKernels.resize(KernelRegistry::GetInstance()->GetKernelCount());

	return true;
}

extern IAdapter* CreateVulkanAdapter()
{
	return AdapterVk::CreateVulkanAdapter();
}

} // End of namespace gpu