#include "gpu/metal/adapter_mtl.h"
#include "gpu/kernel_registry.h"
#include "core/global.h"
#include <iostream>



namespace gpu
{

void* AdapterMtl::Alloc(const size_t size, const AllocationMode mode)
{
	// Initialize GPU if needed
	if (!Init())
	{
		std::cerr << "Failed to initialize GPU" << std::endl;
		return nullptr;
	}

	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// create a new metal buffer that will back the allocation
	const MTL::ResourceOptions options = mode == AllocationMode::Device ? MTL::ResourceStorageModePrivate : MTL::ResourceStorageModeShared;
	NS::SharedPtr<MTL::Buffer> pBuffer = TransferPtr(mpDevice->newBuffer(size, options));
	if (!pBuffer)
	{
		std::cerr << "Failed to create new metal buffer" << std::endl;
		return nullptr;
	}

	// map the allocation to CPU or get its GPU address
	const uint64_t ptr =
		mode == AllocationMode::Device ?
		reinterpret_cast<uint64_t>(pBuffer->gpuAddress()) :
		reinterpret_cast<uint64_t>(pBuffer->contents());

	mAllocations[ptr] = pBuffer;
	return reinterpret_cast<void*>(ptr);
}

void AdapterMtl::Free(void* const ptr)
{
	// remove the allocation record which should also destroy the buffer
	mAllocations.erase(reinterpret_cast<uint64_t>(ptr));
}

bool AdapterMtl::ExecuteKernel(
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

	// an autorelease pool to make sure that any temporary (autorelease) allocations (e.g. the command buffer)
	// get destroyed at the end of this function
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// Get PSO
	const KernelInfo* pKernel = RequestKernel(id);
	if (pKernel == nullptr)
	{
		std::cerr << "Failed to initialize GPU data for kernel " << KernelRegistry::GetInstance()->GetKernelName(id) << std::endl;
		return false;
	}

	// Enqueue GPU commands and execute them
	MTL::CommandBuffer* pCommandBuffer = mpCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	// bind the compute pipeline
	pEncoder->setComputePipelineState(pKernel->pPSO.get());

	// prepare and bind kernel arguments
	if (!EncodeKernelArguments(pEncoder, pKernel, reinterpret_cast<const uint8_t*>(pArgs), pArgsInfo))
	{
		pEncoder->endEncoding();
		std::cerr << "Failed to update arguments for kernel " << KernelRegistry::GetInstance()->GetKernelName(id) << std::endl;
		return false;
	}

	// equeue kernel dispatch
	const MTL::Size gridSize(nx, ny, nz);
	const MTL::Size groupSize(pKernel->pPSO->maxTotalThreadsPerThreadgroup(), 1, 1);  // TODO
	pEncoder->dispatchThreads(gridSize, groupSize);

	// finish encoding and execute computation on the GPU and wait for it
	pEncoder->endEncoding();
	pCommandBuffer->commit();
	pCommandBuffer->waitUntilCompleted();

	// how long the GPU took to execute the command buffer (i.e. the computation kernel)
	const double kernelStartTime = pCommandBuffer->kernelStartTime();
	const double kernelEndTime = pCommandBuffer->kernelEndTime();
	const double kerneTimelMs = (kernelEndTime - kernelStartTime) * 1000.0;
	const double gpuStartTime = pCommandBuffer->GPUStartTime();
	const double gpuEndTime = pCommandBuffer->GPUEndTime();
	const double gpuTimeMs = (gpuEndTime - gpuStartTime) * 1000.0;
	std::cout << "GPU scheduling time: " << kerneTimelMs << " ms" << std::endl;
	std::cout << "GPU execution time: " << gpuTimeMs << " ms" << std::endl;

	return true;
}

bool AdapterMtl::Init(const bool debugMode)
{
	// the command queue is initialized as last in the initialization process,
	// so if we have a valid queue, we know the initialization has already succeeded in the past
	if (mpCommandQueue)
		return true;

	if (debugMode)
	{
		setenv("MTL_DEBUG_LAYER", "1", 1);                  // Enable Metal validation
		setenv("MTL_SHADER_VALIDATION", "1", 1);            // shader-side validation
		setenv("MTL_DEBUG_LAYER_WARNING_MODE", "nslog", 1); // log warnings
	}

	// create an autorelease pool for this adapter
	if (!(mpAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init())))
	{
		std::cerr << "Failed to create a global autorelease pool for the adapter" << std::endl;
		return false;
	}

	// an autorelease pool to make sure that any temporary (autorelease) allocations (e.g. the command buffer)
	// get destroyed at the end of this function
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// create default system device
	if (!(mpDevice = TransferPtr(MTL::CreateSystemDefaultDevice())))
	{
		std::cerr << "Failed to create system default device." << std::endl;;
		return false;
	}

	// load shaders
	if (!(mpLibrary = TransferPtr(mpDevice->newDefaultLibrary())))
	{
		std::cerr << "Failed to load default shader library." << std::endl;;
		return false;
	}

	// create command queue for submitting commands to the GPU
	if (!(mpCommandQueue = TransferPtr(mpDevice->newCommandQueue())))
	{
		std::cerr << "Failed to create command queue" << std::endl;
		return false;
	}

	// ensure the cache is large enough for all kernels
	// all CPU kernels have a static ID variable intialized via kernel registry
	// before main is started, so at this point all kernels
	// should be registered and this should be a valid code
	mKernels.resize(KernelRegistry::GetInstance()->GetKernelCount());

	return true;
}

const AdapterMtl::KernelInfo* AdapterMtl::RequestKernel(const uint32_t id)
{
	EM_ASSERT((mKernels.size() == KernelRegistry::GetInstance()->GetKernelCount()) && "Kernel registry size mismatch");

	const std::string& kernelName = KernelRegistry::GetInstance()->GetKernelName(id);
	if (kernelName.empty())
	{
		std::cerr << "Invalid kernel id " << id << std::endl;
		return nullptr;
	}

	auto& kernel = mKernels[id];
	if (!kernel.pPSO)
	{
		// load the kernel function from shader library
		NS::SharedPtr<MTL::Function> pFunc = TransferPtr(mpLibrary->newFunction(NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding)));
		if (!pFunc)
		{
			std::cerr << "Kernel function " << kernelName << " not found in the shader library" << std::endl;;
			return nullptr;
		}

		// create a compute pipeline for the function, so that it can be executed
		NS::Error* pError = nullptr;
		NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(mpDevice->newComputePipelineState(pFunc.get(), &pError));
		if (!pPSO)
		{
			std::cerr << "Failed to load create compute pipeline: " << pError->localizedDescription()->utf8String() << std::endl;;
			return nullptr;
		}

		// only after all creation succeeded add the entry to the cache
		kernel.pFunc = pFunc;
		kernel.pPSO = pPSO;
	}

	return &kernel;
}

bool AdapterMtl::EncodeKernelArguments(
	MTL::ComputeCommandEncoder* pEncoder,
	const KernelInfo* pKernel,
	const uint8_t* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	// we assume that all kernels take only one argument, which is an argument buffer containing the arguments structure
	NS::SharedPtr<MTL::ArgumentEncoder> pArgEncoder = TransferPtr(pKernel->pFunc->newArgumentEncoder(0));

	// allocate a new argument buffer to store the arguments of the kernel (TODO: replace this with a ring buffer allocator)
	NS::SharedPtr<MTL::Buffer> pArgBuffer = TransferPtr(mpDevice->newBuffer(pArgEncoder->encodedLength(), MTL::ResourceStorageModeShared));

	// tell argument encoder about the buffer where it will serialize kernel arguments
	pArgEncoder->setArgumentBuffer(pArgBuffer.get(), 0);

	// iterate over properties of the arguments structure and fill them up
	if (!EncodeKernelArguments(pEncoder, pArgEncoder.get(), pArgs, pArgsInfo))
		return false;

	// bind the argument buffer
	pEncoder->setBuffer(pArgBuffer.get(), 0, 0);

	return true;
}

bool AdapterMtl::EncodeKernelArguments(
	MTL::ComputeCommandEncoder* pEncoder,
	MTL::ArgumentEncoder* pArgEncoder,
	const uint8_t* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	for (const refl::TypeMetaInfo* info = pArgsInfo; info; info = info->next)
	{
		switch (info->type)
		{
			case refl::TypeTag::Structure:
				if (EncodeKernelArguments(pEncoder, pArgEncoder, pArgs, info->fields))
					break;
				else
					return false;

			case refl::TypeTag::Pointer:
			case refl::TypeTag::ConstPointer:
				if (MTL::Buffer* pBuffer = GetAllocationBuffer(*reinterpret_cast<const uint64_t*>(pArgs + info->offset)))
				{
					static constexpr const NS::UInteger kUsageModeRW = MTL::ResourceUsageRead | MTL::ResourceUsageWrite;
					static constexpr const NS::UInteger kUsageModeR  = MTL::ResourceUsageRead;
					pArgEncoder->setBuffer(pBuffer, 0, info->location);
					pEncoder->useResource(pBuffer, info->type == refl::TypeTag::ConstPointer ? kUsageModeR : kUsageModeRW);
					break;
				}
				else
					return false;

			default:
				std::memcpy(pArgEncoder->constantData(info->location), pArgs + info->offset, info->size);
				break;
		}
	}

	return true;
}

MTL::Buffer* AdapterMtl::GetAllocationBuffer(const void* const ptr) const
{
	return GetAllocationBuffer(reinterpret_cast<uint64_t>(ptr));
}

MTL::Buffer* AdapterMtl::GetAllocationBuffer(const uint64_t ptr) const
{
	const auto it = mAllocations.find(ptr);
	if (it == mAllocations.end())
	{
		std::cerr << "Invalid GPU memory pointer. There is no GPU buffer mapped to address " << ptr << std::endl;
		return nullptr;
	}
	return it->second.get();
}

extern IAdapter* CreateMetalAdapter(const bool debugMode)
{
	return AdapterMtl::CreateMetalAdapter(debugMode);
}

} // End of namespace gpu