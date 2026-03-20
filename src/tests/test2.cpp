#include "core/mtl.h"
#include "kernels/add.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>



static void GenerateData(float* data, const size_t n)
{
	static std::random_device sRngSeed;
	static std::mt19937 sRng(sRngSeed());
	static std::uniform_real_distribution<float> sDist(0.0f, 1000.0f);

	for (size_t i = 0; i < n; ++i)
	{
		data[i] = sDist(sRng);
	}
}

static void AddCPU(const float* a, const float* b, float* c, const size_t n)
{
	for (uint32_t i = 0; i < n; ++i)
	{
		add_kernel(i, a, b, c);
	}
}

static void AddGPU(
	MTL::CommandQueue* pCommandQueue,
	MTL::ComputePipelineState* pPSO,
	const MTL::Buffer* bufA,
	const MTL::Buffer* bufB,
	const MTL::Buffer* bufC,
	const size_t n)
{
	// Enqueue GPU commands
	MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	pEncoder->setComputePipelineState(pPSO);
	pEncoder->setBuffer(bufA, 0, 0);
	pEncoder->setBuffer(bufB, 0, 1);
	pEncoder->setBuffer(bufC, 0, 2);

	MTL::Size gridSize = MTL::Size(n, 1, 1);

	NS::UInteger threadGroupSize = std::min(pPSO->maxTotalThreadsPerThreadgroup(), n);

	MTL::Size threadsPerThreadgroup = MTL::Size(threadGroupSize, 1, 1);

	pEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
	pEncoder->endEncoding();

	// execute computation on the GPU and wait for it
	pCommandBuffer->commit();
	pCommandBuffer->waitUntilCompleted();
}


bool test2()
{
	// a global autorelease pool for all temporary objects released with autorelease (otherwise they would leak)
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// create default system device
	NS::SharedPtr<MTL::Device> pDevice = TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!pDevice)
	{
		std::cerr << "Failed to create system default device." << std::endl;;
		return false;
	}

	// load shaders
	NS::SharedPtr<MTL::Library> pLibrary = TransferPtr(pDevice->newDefaultLibrary());
	if (!pLibrary)
	{
		std::cerr << "Failed to load default library." << std::endl;;
		return false;
	}

	// create compute pipeline for the add_kernel shader
	NS::Error* pError = nullptr;
	NS::SharedPtr<MTL::Function> pAddFn = TransferPtr(pLibrary->newFunction(NS::String::string("add_kernel", NS::UTF8StringEncoding)));
	NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(pDevice->newComputePipelineState(pAddFn.get(), &pError));
	if (!pPSO)
	{
		std::cerr << "Failed to load create compute pipeline: " << pError->localizedDescription()->utf8String() << std::endl;;
		return false;
	}

	// create command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> pCommandQueue = TransferPtr(pDevice->newCommandQueue());
	if (!pCommandQueue)
	{
		std::cerr << "Failed to create command queue" << std::endl;
		return false;
	}

	// Create buffers
	const size_t N = 100000000;

	NS::SharedPtr<MTL::Buffer> bufA = TransferPtr(pDevice->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> bufB = TransferPtr(pDevice->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resCPU = TransferPtr(pDevice->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resGPU = TransferPtr(pDevice->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared));

	// Create input data
	GenerateData(static_cast<float*>(bufA->contents()), N);
	GenerateData(static_cast<float*>(bufB->contents()), N);

	// compute on CPU (reference)
	auto cpu_t0 = std::chrono::high_resolution_clock::now();
	AddCPU(static_cast<const float*>(bufA->contents()), static_cast<const float*>(bufB->contents()), static_cast<float*>(resCPU->contents()), N);
	auto cpu_t1 = std::chrono::high_resolution_clock::now();

	// compute on GPU
	auto gpu_t0 = std::chrono::high_resolution_clock::now();
	AddGPU(pCommandQueue.get(), pPSO.get(), bufA.get(), bufB.get(), resGPU.get(), N);
	auto gpu_t1 = std::chrono::high_resolution_clock::now();

	// validate results
	const float* cpu = static_cast<const float*>(resCPU->contents());
	const float* gpu = static_cast<const float*>(resGPU->contents());
	for (size_t i = 0; i < N; ++i)
	{
		if (cpu[i] != gpu[i])
		{
			std::cerr << "Invalid results, cpu and gpu differ at element " << i << ": " << cpu[i] << " vs " << gpu[i] << std::endl;
			return false;
		}
	}

	std::cout << "CPU and GPU produces equivalent outputs" << std::endl;
	std::cout << "CPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_t1 - cpu_t0).count() << " ms" << std::endl;
	std::cout << "GPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(gpu_t1 - gpu_t0).count() << " ms" << std::endl;

	return true;
}
