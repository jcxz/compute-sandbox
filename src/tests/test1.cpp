#include "core/mtl.h"
#include "kernels/add.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>



static std::vector<float> GenerateData(const size_t n)
{
	static std::random_device sRngSeed;
	static std::mt19937 sRng(sRngSeed());
	static std::uniform_real_distribution<float> sDist(0.0f, 1000.0f);

	std::vector<float> out;
	for (size_t i = 0; i < n; ++i)
	{
		out.push_back(sDist(sRng));
	}

	return out;
}

static bool AddCPU(const float* a, const float* b, float* c, const size_t n)
{
	for (uint32_t i = 0; i < n; ++i)
	{
		add_kernel(i, a, b, c);
	}
	return true;
}

static bool AddGPU(const float* a, const float* b, float* c, const size_t n)
{
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
	NS::SharedPtr<MTL::Buffer> buf1 = TransferPtr(pDevice->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> buf2 = TransferPtr(pDevice->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> result = TransferPtr(pDevice->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared));

	std::memcpy(buf1->contents(), a, n * sizeof(float));
	std::memcpy(buf2->contents(), b, n * sizeof(float));

	// Enqueue GPU commands
	MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	pEncoder->setComputePipelineState(pPSO.get());
	pEncoder->setBuffer(buf1.get(), 0, 0);
	pEncoder->setBuffer(buf2.get(), 0, 1);
	pEncoder->setBuffer(result.get(), 0, 2);

	MTL::Size gridSize = MTL::Size(n, 1, 1);

	NS::UInteger threadGroupSize = std::min(pPSO->maxTotalThreadsPerThreadgroup(), n);

	MTL::Size threadsPerThreadgroup = MTL::Size(threadGroupSize, 1, 1);

	pEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
	pEncoder->endEncoding();

	// execute computation on the GPU and wait for it
	pCommandBuffer->commit();
	pCommandBuffer->waitUntilCompleted();

	// Get results
	std::memcpy(c, static_cast<const float*>(result->contents()), n * sizeof(float));

	return true;
}


bool test1()
{
	// a global autorelease pool for all temporary objects released with autorelease (otherwise they would leak)
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// Create input data
	const size_t N = 100000000;
	std::vector<float> a = GenerateData(N);
	std::vector<float> b = GenerateData(N);

	// compute on CPU (reference)
	std::vector<float> cpu(N);
	auto cpu_t0 = std::chrono::high_resolution_clock::now();
	if (!AddCPU(a.data(), b.data(), cpu.data(), N))
	{
		std::cerr << "CPU computation failed" << std::endl;
		return false;
	}
	auto cpu_t1 = std::chrono::high_resolution_clock::now();

	// compute on GPU
	std::vector<float> gpu(N);
	auto gpu_t0 = std::chrono::high_resolution_clock::now();
	if (!AddGPU(a.data(), b.data(), gpu.data(), N))
	{
		std::cerr << "GPU computation failed" << std::endl;
		return false;
	}
	auto gpu_t1 = std::chrono::high_resolution_clock::now();

	// validate results
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
