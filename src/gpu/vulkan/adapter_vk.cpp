#include "gpu/vulkan/adapter_vk.h"
#include "gpu/kernel_registry.h"
#include <iostream>
#include <cstring>



static void ReflectOffsets(slang::TypeLayoutReflection* pTypeLayout, size_t baseOffset, std::vector<size_t>& offsets)
{
	const uint32_t fieldCount = pTypeLayout->getFieldCount();
	for (uint32_t i = 0; i < fieldCount; ++i)
	{
		// store the offset of the current field
		slang::VariableLayoutReflection* pField = pTypeLayout->getFieldByIndex(i);
		offsets.push_back(baseOffset + pField->getOffset());

		std::cout << "Reflecting field " << pField->getName() << " at offset " << baseOffset + pField->getOffset() << std::endl;

		// if it's a structure, we need to recursively reflect its member offsets
		slang::TypeLayoutReflection* pFieldType = pField->getTypeLayout();
		if (pFieldType->getKind() == slang::TypeReflection::Kind::Struct)
		{
			ReflectOffsets(pFieldType, baseOffset + offsets.back(), offsets);
		}
	}
}

static void ReflectOffsets(slang::TypeLayoutReflection* pTypeLayout, std::vector<size_t>& offsets, size_t& size)
{
	std::cout << __FUNCTION__
			  << " name=" << (pTypeLayout->getName() ? pTypeLayout->getName() : "null")
			  << " kind=" << (int)pTypeLayout->getKind()
			  << " size=" << pTypeLayout->getSize() << std::endl;

	size = pTypeLayout->getSize();
	if (pTypeLayout->getKind() == slang::TypeReflection::Kind::Struct)
	{
		ReflectOffsets(pTypeLayout, 0, offsets);
	}
	else
	{
		offsets.push_back(0);
	}
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	(void)messageSeverity;
	(void)messageType;
	(void)pUserData;
	std::cerr << "Vulkan: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
		func(instance, debugMessenger, pAllocator);
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

	if (mPipelineCache)
		vkDestroyPipelineCache(mDevice, mPipelineCache, nullptr);

	if (mDevice)
		vkDestroyDevice(mDevice, nullptr);

	if (mDebugMessenger)
		DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);

	if (mInstance)
		vkDestroyInstance(mInstance, nullptr);
}

void* AdapterVk::Alloc(const size_t size, const AllocationMode mode)
{
	EM_ASSERT(IsInitialized() && "Vulkan adapter must be initialized prior to calling any of its functions");

	Allocation alloc(this);

	// Create buffer
	{
		VkBufferCreateInfo bufferInfo;
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.pNext = nullptr;
		bufferInfo.flags = 0;
		bufferInfo.size = size;
		bufferInfo.usage =
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
			VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR;  // so that we can query the GPU device address of the buffer
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferInfo.queueFamilyIndexCount = 0;
		bufferInfo.pQueueFamilyIndices = nullptr;

		const VkResult res = vkCreateBuffer(mDevice, &bufferInfo, nullptr, &alloc.buffer);
		if (res != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::Alloc failed: Failed to create backing Vulkan buffer (" << VkResultToString(res) << ")" << std::endl;
			return nullptr;
		}
	}

	// Allocate memory
	{
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(mDevice, alloc.buffer, &memRequirements);

		const VkMemoryPropertyFlags properties =
			(mode == AllocationMode::Device) ?
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT :
			(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VkMemoryAllocateFlagsInfo allocateFlagsInfo;
		allocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		allocateFlagsInfo.pNext = nullptr;
		allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
		allocateFlagsInfo.deviceMask = 0;

		VkMemoryAllocateInfo allocInfo;
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext = &allocateFlagsInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
		if (allocInfo.memoryTypeIndex == uint32_t(-1))
		{
			std::cerr << "AdapterVk::Alloc failed: Failed to find suitable memory type." << std::endl;
			return nullptr;
		}

		const VkResult res = vkAllocateMemory(mDevice, &allocInfo, nullptr, &alloc.memory);
		if (res != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::Alloc failed: Failed to allocate buffer memory for the backing Vulkan buffer (" << VkResultToString(res) << ")" << std::endl;
			return nullptr;
		}
	}

	// bind the memory allocation to the buffer
	{
		const VkResult res = vkBindBufferMemory(mDevice, alloc.buffer, alloc.memory, 0);
		if (res != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::Alloc: Failed to bind memory to the backing Vulkan buffer (" << VkResultToString(res) << ")" << std::endl;
			return nullptr;
		}
	}

	// get the device address of the buffer
	{
		VkBufferDeviceAddressInfo bufferDeviceAddressInfo;
		bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		bufferDeviceAddressInfo.pNext = nullptr;
		bufferDeviceAddressInfo.buffer = alloc.buffer;
		alloc.deviceAddress = vkGetBufferDeviceAddress(mDevice, &bufferDeviceAddressInfo);
		if (alloc.deviceAddress == 0)
		{
			std::cerr << "AdapterVk::Alloc: Failed to get the device address of the backing Vulkan buffer" << std::endl;
			return nullptr;
		}
	}

	// Map memory
	if (mode == AllocationMode::Shared)
	{
		const VkResult res = vkMapMemory(mDevice, alloc.memory, 0, size, 0, &alloc.ptr);
		if (res != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::Alloc failed: Failed to map the buffer for access by the CPU (" << VkResultToString(res) << ")" << std::endl;
			return nullptr;
		}
	}

	// Treat the buffer handle as the pointer reference for purely device local memory
	// if we aren't mapping it explicitly. For a shared abstraction, we need to return
	// something unique.
	void* ptr = (mode == AllocationMode::Shared) ? alloc.ptr : reinterpret_cast<void*>(alloc.deviceAddress);

	mAllocations[ptr] = std::move(alloc);
	return ptr;
}

void AdapterVk::Free(void* const ptr)
{
	EM_ASSERT(IsInitialized() && "Vulkan adapter must be initialized prior to calling any of its functions");
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
	EM_ASSERT(IsInitialized() && "Vulkan adapter must be initialized prior to calling any of its functions");

	// 1. Get/Build Pipeline
	const KernelInfo* pKernel = RequestKernel(id);
	if (pKernel == nullptr)
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Failed to initialize GPU data for kernel "
				  << KernelRegistry::GetInstance()->GetKernelName(id) << " (id: " << id << ")" << std::endl;
		return false;
	}

	// 2. fill the arguments GPU buffer with kernel's arguments
	if (!EncodeKernelArguments(pKernel, static_cast<const uint8_t*>(pArgs), pArgsInfo))
	{
		std::cerr << "AdapterVk::ExecuteKernel() - Failed to encode kernel arguments!" << std::endl;
		return false;
	}

	// 3. Allocate a command buffer
	VkCommandBuffer commandBuffer;
	{
		VkCommandBufferAllocateInfo commandBufferAllocInfo;
		commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocInfo.pNext = nullptr;
		commandBufferAllocInfo.commandPool = mCommandPool;
		commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocInfo.commandBufferCount = 1;

		const VkResult res = vkAllocateCommandBuffers(mDevice, &commandBufferAllocInfo, &commandBuffer);
		if (res != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::ExecuteKernel() - Failed to allocate command buffers!" << std::endl;
			return false;
		}
	}

	// 4. Begin recording
	{
		VkCommandBufferBeginInfo beginInfo;
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pNext = nullptr;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		const VkResult res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		if (res != VK_SUCCESS)
		{
			std::cerr << "vkBeginCommandBuffer failed with " << VkResultToString(res) << std::endl;
			vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
			return false;
		}
	}

	// 5. Bind Pipeline and push constants
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pKernel->pipeline);
	vkCmdPushConstants(commandBuffer, pKernel->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress), &pKernel->argsBufferDeviceAddress);

	// 6. Dispatch
	vkCmdDispatch(commandBuffer, nx, ny, nz);

	{
		const VkResult res = vkEndCommandBuffer(commandBuffer);
		if (res != VK_SUCCESS)
		{
			std::cerr << "vkEndCommandBuffer failed with " << VkResultToString(res) << std::endl;
			vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
			return false;
		}
	}

	// 7. Submit
	{
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

		const VkResult res = vkQueueSubmit(mComputeQueue, 1, &submitInfo, VK_NULL_HANDLE);
		if (res != VK_SUCCESS)
		{
			std::cerr << "vkQueueSubmit failed with " << VkResultToString(res) << std::endl;
			vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
			return false;
		}
	}


	// 8. Wait (Synchronous for now)
	{
		const VkResult res = vkQueueWaitIdle(mComputeQueue);
		if (res != VK_SUCCESS)
		{
			std::cerr << "vkQueueWaitIdle failed with " << VkResultToString(res) << std::endl;
			vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
			return false;
		}
	}

	// 9. Cleanup
	vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);

	return true;
}

uint32_t AdapterVk::FindMemoryType(const uint32_t typeFilter, const VkMemoryPropertyFlags properties) const
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
	return uint32_t(-1);
}

const AdapterVk::KernelInfo* AdapterVk::RequestKernel(const uint32_t id)
{
	EM_ASSERT((mKernels.size() == KernelRegistry::GetInstance()->GetKernelCount()) && "Kernel registry size mismatch");

	// first make sure that the provided kernel id is valid
	const std::string& kernelName = KernelRegistry::GetInstance()->GetKernelName(id);
	if (kernelName.empty())
	{
		std::cerr << "Invalid kernel id " << id << std::endl;
		return nullptr;
	}

	// then look it up in the cache and set it up if necessary
	auto& kernel = mKernels[id];
	if (kernel.pipeline == VK_NULL_HANDLE)
	{
		KernelInfo kernelInfo(this);

		// build slang program
		VkShaderModule shaderModule;
		Slang::ComPtr<slang::IComponentType> pProgram;
		if (!BuildSlangProgram(kernelName, pProgram.writeRef(), shaderModule))
		{
			std::cerr << "Failed to compile " << kernelName << std::endl;
			return nullptr;
		}

		// perform program's reflection to know how to pass arguments to the kernel at runtime
		if (!ReflectSlangProgram(pProgram, kernelInfo.reflectionInfo))
		{
			std::cerr << "Failed to reflect " << kernelName << std::endl;
			vkDestroyShaderModule(mDevice, shaderModule, nullptr);
			return nullptr;
		}

		// create pipeline layout
		{
			VkPushConstantRange pushConstantRange;
			pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			pushConstantRange.offset = 0;
			pushConstantRange.size = sizeof(VkDeviceAddress);

			VkPipelineLayoutCreateInfo pipelineLayoutInfo;
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.pNext = nullptr;
			pipelineLayoutInfo.flags = 0;
			pipelineLayoutInfo.setLayoutCount = 0;
			pipelineLayoutInfo.pSetLayouts = nullptr;
			pipelineLayoutInfo.pushConstantRangeCount = 1;
			pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

			const VkResult res = vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &kernelInfo.pipelineLayout);
			if (res != VK_SUCCESS)
			{
				std::cerr << "Failed to create VkPipelineLayout: " << VkResultToString(res) << std::endl;
				vkDestroyShaderModule(mDevice, shaderModule, nullptr);
				return VK_NULL_HANDLE;
			}
		}

		// create pipeline
		{
			VkComputePipelineCreateInfo pipelineInfo;
			pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineInfo.pNext = nullptr;
			pipelineInfo.flags = 0;
			pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			pipelineInfo.stage.pNext = nullptr;
			pipelineInfo.stage.flags = 0;
			pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			pipelineInfo.stage.module = shaderModule;
			pipelineInfo.stage.pName = "main";
			pipelineInfo.stage.pSpecializationInfo = nullptr;
			pipelineInfo.layout = kernelInfo.pipelineLayout;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
			pipelineInfo.basePipelineIndex = -1;

			const VkResult res = vkCreateComputePipelines(mDevice, mPipelineCache, 1, &pipelineInfo, nullptr, &kernelInfo.pipeline);
			if (res != VK_SUCCESS)
			{
				std::cerr << "Failed to create compute pipeline for " << kernelName << " (" << VkResultToString(res) << ")" << std::endl;
				vkDestroyShaderModule(mDevice, shaderModule, nullptr);
				return nullptr;
			}
		}

		// now that the pipeline was created we can delete the shader module
		vkDestroyShaderModule(mDevice, shaderModule, nullptr);

		// allocate buffer for kernel arguments
		kernelInfo.pArgsBuffer = Alloc(kernelInfo.reflectionInfo.argsBufferSize, AllocationMode::Shared);
		if (kernelInfo.pArgsBuffer == nullptr)
		{
			std::cerr << "Failed to preallocate memory for arguments of kernel " << kernelName << std::endl;
			return nullptr;
		}

		kernelInfo.argsBufferDeviceAddress = GetAllocation(kernelInfo.pArgsBuffer)->deviceAddress;

		// now that everyhting succeeded we can finally store all information to the kernel cache
		kernel = std::move(kernelInfo);
	}

	return &kernel;
}

bool AdapterVk::BuildSlangProgram(const std::string& kernelName, slang::IComponentType** ppProgram, VkShaderModule& shaderModule)
{
	// compile slang shader
	Slang::ComPtr<slang::IModule> pModule;
	{
		Slang::ComPtr<slang::IBlob> pDiagnostics;
		pModule = mSlangSession->loadModule(kernelName.c_str(), pDiagnostics.writeRef());
		if (pDiagnostics)
			std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
		if (!pModule)
		{
			std::cerr << "Failed to load compile shader module for kernel " << kernelName << std::endl;
			return false;
		}
	}

	Slang::ComPtr<slang::IEntryPoint> pEntryPoint;
	if (pModule->findEntryPointByName(kernelName.c_str(), pEntryPoint.writeRef()) != SLANG_OK)
	{
		std::cerr << "Failed to find entry point for kernel " << kernelName << std::endl;
		return false;
	}

	// link slang shader
	{
		slang::IComponentType* components[] = { pModule, pEntryPoint };
		Slang::ComPtr<slang::IBlob> pDiagnostics;
		SlangResult result = mSlangSession->createCompositeComponentType(components, SLANG_COUNT_OF(components), ppProgram, pDiagnostics.writeRef());
		if (pDiagnostics)
			std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
		if (result != SLANG_OK)
		{
			std::cerr << "Failed to create link shader program for kernel " << kernelName << std::endl;
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

	Slang::ComPtr<slang::IBlob> pSPIRVCode;
	{
		Slang::ComPtr<slang::IBlob> pDiagnostics;
		const SlangResult result = (*ppProgram)->getEntryPointCode(0, 0, pSPIRVCode.writeRef(), pDiagnostics.writeRef());
		if (pDiagnostics)
			std::cerr << (const char*)pDiagnostics->getBufferPointer() << std::endl;
		if (result != SLANG_OK)
		{
			std::cerr << "Failed to get SPIR-V code from Slang program." << std::endl;
			return false;
		}
	}

	{
		VkShaderModuleCreateInfo moduleCreateInfo;
		moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleCreateInfo.pNext = nullptr;
		moduleCreateInfo.flags = 0;
		moduleCreateInfo.codeSize = pSPIRVCode->getBufferSize();
		moduleCreateInfo.pCode = (const uint32_t*)pSPIRVCode->getBufferPointer();

		const VkResult res = vkCreateShaderModule(mDevice, &moduleCreateInfo, nullptr, &shaderModule);
		if (res != VK_SUCCESS)
		{
			std::cerr << "Failed to create shader VkShaderModule for kernel " << kernelName << "(" << VkResultToString(res) << ")" << std::endl;
			return false;
		}
	}

	return true;
}

bool AdapterVk::ReflectSlangProgram(slang::IComponentType* pProgram, KernelReflectionInfo& reflectionInfo)
{
	std::cout << __FUNCTION__ << std::endl;

	slang::ProgramLayout* pReflection = pProgram->getLayout();
	slang::EntryPointReflection* pEntryPointReflection = pReflection->getEntryPointByIndex(0);

	if (pEntryPointReflection->getStage() != SlangStage::SLANG_STAGE_COMPUTE)
	{
		std::cerr << pEntryPointReflection->getName() << " is not a compute kernel." << std::endl;
		return false;
	}

	const uint32_t parameterCount = pEntryPointReflection->getParameterCount();
	if (parameterCount != 2)
	{
		std::cerr << pEntryPointReflection->getName() << " is likley not a valid kernel. Typical kernels contain exactly 2 parameters." << std::endl;
		return false;
	}

	size_t offset = 0;
	size_t size = 0;
	for (uint32_t i = 0; i < parameterCount; ++i)
	{
		slang::VariableLayoutReflection* pParam = pEntryPointReflection->getParameterByIndex(i);
		slang::TypeLayoutReflection* pParamType = pParam->getTypeLayout();
		//std::cout << "param name=" << (pParam->getName() ? pParam->getName() : "null") << " kind=" << (int)pParamType->getKind() << " name=" << pParamType->getName() << std::endl;
		switch (pParam->getCategory())
		{
			case slang::ParameterCategory::Uniform:
				if (pParamType->getKind() != slang::TypeReflection::Kind::Pointer)
				{
					std::cerr << "Expected " << pParam->getName() << " parameter to be a uniform pointer." << std::endl;
					return false;
				}
				offset = pParam->getOffset(slang::ParameterCategory::Uniform);
				size = pParamType->getSize(slang::ParameterCategory::Uniform);
				ReflectOffsets(pParamType->getElementTypeLayout(), reflectionInfo.offsets, reflectionInfo.argsBufferSize);
				break;

			case slang::ParameterCategory::None:
				if (std::strcmp(pParam->getSemanticName(), "SV_DISPATCHTHREADID") != 0)
				{
					std::cerr << "Expected an SV_DispatchThreadID value semantics on parameter " << pParam->getName() << std::endl;
					return false;
				}
				break;

			default:
				std::cerr << "Unexpected parameter of category: " << pParam->getCategory() << std::endl;
				return false;
		}
	}

	if (offset != 0)
	{
		std::cerr << "Invalid kernel signature, argument offset != 0." << std::endl;
		return false;
	}

	if (size != sizeof(VkDeviceAddress))
	{
		std::cerr << "Invalid kernel signature, likely missing a uniform pointer to kernel arguments." << std::endl;
		return false;
	}

	pEntryPointReflection->getComputeThreadGroupSize(3, reflectionInfo.threadGroupSize);

	return true;
}

bool AdapterVk::IsInitialized() const
{
	// the descriptor pool is initialized as last in the initialization process,
	// so if we have a valid descriptor pool, we know the initialization has already succeeded in the past
	return mDescriptorPool != VK_NULL_HANDLE;
}

bool AdapterVk::Init(const bool debugMode)
{
	static const char* const kIncludePaths[] = {
		"D:/programovanie/compute-sandbox/src",
		"D:/programovanie/compute-sandbox/src/tests/invert"
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

	// enable validation layers and debug utils extension
	static const char* const validationLayers[] =
	{
		"VK_LAYER_KHRONOS_validation"
	};

	static const char* const extensions[] =
	{
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME
	};

	// 2. Create Instance
	VkApplicationInfo appInfo;
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pNext = nullptr;
	appInfo.pApplicationName = "";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	// we default to Vulkan 1.3 because we rely on buffer device address feature,
	// which is widely supported by HW, but core to the vulkan specification only since version 1.3
	// for more info see also this tutorial: https://howtovulkan.com/#shader-data-buffers
	appInfo.apiVersion = VK_API_VERSION_1_3;

	VkInstanceCreateInfo createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pNext = nullptr;
	createInfo.flags = 0;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledLayerCount = debugMode ? 1 : 0;
	createInfo.ppEnabledLayerNames = validationLayers;
	createInfo.enabledExtensionCount = debugMode ? 1 : 0;
	createInfo.ppEnabledExtensionNames = extensions;

	if (vkCreateInstance(&createInfo, nullptr, &mInstance) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create Vulkan instance!" << std::endl;
		return false;
	}

	// Setup debug messenger
	if (debugMode)
	{
		VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo;
		debugMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugMessengerCreateInfo.pNext = nullptr;
		debugMessengerCreateInfo.flags = 0;
		debugMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugMessengerCreateInfo.pfnUserCallback = &DebugUtilsCallback;
		debugMessengerCreateInfo.pUserData = nullptr;

		if (CreateDebugUtilsMessengerEXT(mInstance, &debugMessengerCreateInfo, nullptr, &mDebugMessenger) != VK_SUCCESS)
		{
			std::cerr << "AdapterVk::Init() - Failed to setup debug messenger!" << std::endl;
		}
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

	//VkPhysicalDeviceFeatures deviceFeatures;
	//vkGetPhysicalDeviceFeatures(mPhysicalDevice, &deviceFeatures);

	VkPhysicalDeviceVulkan11Features vk11Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = nullptr
	};

	VkPhysicalDeviceVulkan12Features vk12Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &vk11Features,
	};

	VkPhysicalDeviceVulkan13Features vk13Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &vk12Features,
	};

	VkPhysicalDeviceFeatures2 deviceFeatures2;
	deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	deviceFeatures2.pNext = &vk13Features;
	vkGetPhysicalDeviceFeatures2(mPhysicalDevice, &deviceFeatures2);

	if (!vk12Features.storagePushConstant8 || !vk12Features.shaderInt8 || !vk12Features.bufferDeviceAddress)
	{
		std::cerr << "AdapterVk::Init() - Required Vulkan 1.2 features are not supported!" << std::endl;
		return false;
	}

	// in modern Vulkan it is allegedly not considered best practive to enable
	// to all supported feature, because certain combinations of features and extensions
	// are not allowed or some newer features may change how features in older Vulkan versions behaved
	// in subtle ways (according to GPT).
	// Also allowing some features may incur certain overhead (e.g. robust image\buffer features)
	// see for example: https://stackoverflow.com/questions/63020989/create-vulkan-instance-device-with-all-supported-extensions-features-enabled
	VkPhysicalDeviceVulkan11Features enabledVk11Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		//.pNext = &deviceFeatures2
	};

	VkPhysicalDeviceVulkan12Features enabledVk12Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &enabledVk11Features,
		//.descriptorIndexing = true,
		//.shaderSampledImageArrayNonUniformIndexing = true,
		//.descriptorBindingVariableDescriptorCount = true,
		//.runtimeDescriptorArray = true,
		.storagePushConstant8 = true,
		.shaderInt8 = true,
		.bufferDeviceAddress = true
	};

	VkPhysicalDeviceVulkan13Features enabledVk13Features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &enabledVk12Features,
		//.synchronization2 = true,
		//.dynamicRendering = true
	};

	// 4. Find Compute Queue Family
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, queueFamilies.data());

	uint32_t computeQueueFamilyIndex = uint32_t(-1);
	for (uint32_t i = 0; i < queueFamilyCount; ++i)
	{
		if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			computeQueueFamilyIndex = i;
			break;
		}
	}

	if (computeQueueFamilyIndex == uint32_t(-1))
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
	deviceCreateInfo.pNext = &enabledVk13Features;
	deviceCreateInfo.flags = 0;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.enabledLayerCount = 0;
	deviceCreateInfo.ppEnabledLayerNames = nullptr;
	deviceCreateInfo.enabledExtensionCount = 0;
	deviceCreateInfo.ppEnabledExtensionNames = nullptr;
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures2.features;   // lets enable all supported device features

	if (vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create logical device!" << std::endl;
		return false;
	}

	vkGetDeviceQueue(mDevice, computeQueueFamilyIndex, 0, &mComputeQueue);

	// 6. create pipeline cache to speed up creating compute pipelines for kernels
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo;
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	pipelineCacheCreateInfo.pNext = nullptr;
	pipelineCacheCreateInfo.flags = 0;
	// if I wanted to load a stored pipeline cache from disk
	pipelineCacheCreateInfo.initialDataSize = 0;
	pipelineCacheCreateInfo.pInitialData = nullptr;

	if (vkCreatePipelineCache(mDevice, &pipelineCacheCreateInfo, nullptr, &mPipelineCache) != VK_SUCCESS)
	{
		std::cerr << "AdapterVk::Init() - Failed to create pipeline cache!" << std::endl;
		return false;
	}

	// 7. Create Command Pool
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

	// 8. Create Descriptor Pool
	const VkDescriptorPoolSize poolSizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024 },
	};

	VkDescriptorPoolCreateInfo descriptorPoolInfo;
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.pNext = nullptr;
	descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // so that we can call vkFreeDescriptorSets
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

bool AdapterVk::EncodeKernelArguments(const KernelInfo* const pKernel, const uint8_t* const pArgs, const refl::TypeMetaInfo* const pArgsInfo)
{
	for (const refl::TypeMetaInfo* info = pArgsInfo; info; info = info->next)
	{
		switch (info->type)
		{
			case refl::TypeTag::Structure:
				if (EncodeKernelArguments(pKernel, pArgs, info->fields))
					break;
				else
					return false;

			case refl::TypeTag::Pointer:
			case refl::TypeTag::ConstPointer:
				if (const Allocation* pAlloc = GetAllocation(*reinterpret_cast<void * const *>(pArgs + info->offset)))
				{
					std::memcpy(static_cast<uint8_t*>(pKernel->pArgsBuffer) + pKernel->reflectionInfo.offsets[info->location], &pAlloc->deviceAddress, sizeof(VkDeviceAddress));
					break;
				}
				else
					return false;

			default:
				std::memcpy(static_cast<uint8_t*>(pKernel->pArgsBuffer) + pKernel->reflectionInfo.offsets[info->location], pArgs + info->offset, info->size);
				break;
		}
	}

	return true;
}

const AdapterVk::Allocation* AdapterVk::GetAllocation(void* const ptr) const
{
	const auto it = mAllocations.find(ptr);
	if (it == mAllocations.end())
	{
		std::cerr << "Invalid GPU memory pointer. There is no GPU buffer mapped to address " << ptr << std::endl;
		return nullptr;
	}
	return &it->second;
}

extern IAdapter* CreateVulkanAdapter(const bool debugMode)
{
	return AdapterVk::CreateVulkanAdapter(debugMode);
}

} // End of namespace gpu