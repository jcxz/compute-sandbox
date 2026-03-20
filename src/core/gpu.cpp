#include "core/gpu.h"
#include "core/reflection.h"
#include "core/mtl.h"

#include <unordered_map>
#include <string>
#include <cassert>
#include <iostream>



namespace
{

class Gpu
{
	static_assert(sizeof(MTL::GPUAddress) == sizeof(uint64_t), "MTL::GPUAddress not the same size as uint64_t");
	static_assert(sizeof(void*) == sizeof(uint64_t), "void* not the same size as uint64_t");

private:
	struct KernelInfo
	{
		std::string name;
		NS::SharedPtr<MTL::Function> pFunc;
		NS::SharedPtr<MTL::ComputePipelineState> pPSO;
		KernelInfo() = default;
		explicit KernelInfo(const std::string& name)
			: name(name)
			, pFunc(nullptr)
			, pPSO(nullptr)
		{ }
	};

public:
	void* Alloc(const size_t size, const AllocationMode mode)
	{
		NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

		// Initialize GPU if needed
		if (!Init())
		{
			std::cerr << "Failed to initialize GPU" << std::endl;
			return nullptr;
		}

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

	void Free(void* const ptr)
	{
		// remove the allocation record which should also destroy the buffer
		mAllocations.erase(reinterpret_cast<uint64_t>(ptr));
	}

	uint32_t RegisterKernel(const std::string& name)
	{
		const uint32_t id = static_cast<uint32_t>(mKernels.size());
		mKernels.emplace_back(name);
		return id;
	}

	bool ExecuteKernel(
		const uint32_t id,
		const uint32_t nx,
		const uint32_t ny,
		const uint32_t nz,
		const void* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo)
	{
		// an autorelease pool to make sure that any temporary (autorelease) allocations (e.g. the command buffer)
		// get destroyed at the end of this function
		NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

		// Initialize GPU if needed
		if (!Init())
		{
			std::cerr << "Failed to initialize GPU" << std::endl;
			return false;
		}

		// Get PSO
		const KernelInfo* pKernel = RequestKernel(id);
		if (pKernel == nullptr)
		{
			std::cerr << "Failed to initialize GPU data for kernel " << pKernel->name << std::endl;
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
			std::cerr << "Failed to update arguments for kernel " << pKernel->name << std::endl;
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

	static Gpu* GetInstance()
	{
		static Gpu gpu;
		return &gpu;
	}

private:
	bool Init()
	{
		// the command queue is initialized as last in the initialization process,
		// so if we have a valid queue, we know the initialization has already succeeded in the past
		if (mpCommandQueue)
			return true;

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

		return true;
	}

	const KernelInfo* RequestKernel(const uint32_t id)
	{
		if (id >= mKernels.size())
		{
			std::cerr << "Invalid kernel id " << id << std::endl;
			return nullptr;
		}
	
		auto& kernel = mKernels[id];
		if (!kernel.pPSO)
		{
			// load the kernel function from shader library
			NS::SharedPtr<MTL::Function> pFunc = TransferPtr(mpLibrary->newFunction(NS::String::string(kernel.name.c_str(), NS::UTF8StringEncoding)));
			if (!pFunc)
			{
				std::cerr << "Kernel function " << kernel.name << " not found in the shader library" << std::endl;;
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

	bool EncodeKernelArguments(
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

	bool EncodeKernelArguments(
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

	MTL::Buffer* GetAllocationBuffer(const void* const ptr) const
	{
		return GetAllocationBuffer(reinterpret_cast<uint64_t>(ptr));
	}

	MTL::Buffer* GetAllocationBuffer(const uint64_t ptr) const
	{
		const auto it = mAllocations.find(ptr);
		if (it == mAllocations.end())
		{
			std::cerr << "Invalid GPU memory pointer. There is no GPU buffer mapped to address " << ptr << std::endl;
			return nullptr;
		}
		return it->second.get();
	}

private:
	Gpu() = default;
	Gpu(Gpu&& ) = delete;
	Gpu(const Gpu& ) = delete;
	Gpu& operator=(Gpu&& ) = delete;
	Gpu& operator=(const Gpu& ) = delete;

private:
	//! the GPU device that will be used to execute kernels
	NS::SharedPtr<MTL::Device> mpDevice;
	//! shader library with the compile kernels
	// (allows us on demand compute pipeline creation)
	NS::SharedPtr<MTL::Library> mpLibrary;
	//! a command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> mpCommandQueue;
	//! a cache of precompiled pipelines ready to be used for starting a kernel
	std::vector<KernelInfo> mKernels;
	//! argument buffer allocator (here we will store the arguments that we pass to kernels)
	//ArgumentBufferAllocator mAllocator; // ... TODO
	//! Let's keep it simple for now, stores references to allocated GPU buffers
	std::unordered_map<uint64_t, NS::SharedPtr<MTL::Buffer>> mAllocations;
};

} // End of private namespace

void* GpuAlloc(const size_t size, const AllocationMode mode)
{
	return Gpu::GetInstance()->Alloc(size, mode);
}

void GpuFree(void* const ptr)
{
	return Gpu::GetInstance()->Free(ptr);
}

uint32_t RegisterKernel(const std::string& name)
{
	return Gpu::GetInstance()->RegisterKernel(name);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return Gpu::GetInstance()->ExecuteKernel(id, nx, 1, 1, pArgs, pArgsInfo);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return Gpu::GetInstance()->ExecuteKernel(id, nx, ny, 1, pArgs, pArgsInfo);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const uint32_t nz,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return Gpu::GetInstance()->ExecuteKernel(id, nx, ny, nz, pArgs, pArgsInfo);
}
