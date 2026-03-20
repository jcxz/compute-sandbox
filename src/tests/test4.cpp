#include "core/mtl.h"
#include "kernels/invert2.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>


static bool WritePPM(
	const char* const path,
	const uint8_t* const pData,
	const size_t strideBytes,
	const uint32_t width,
	const uint32_t height)
{
	FILE* f = fopen(path, "w");
	if (f == nullptr)
	{
		std::cerr << "Failed to open " << path << " for writing" << std::endl;
		return false;
	}

	fprintf(f, "%s\n%d %d\n255\n", "P2", width, height);

	for (uint32_t y = 0; y < height; y++)
	{
		const uint8_t* pLine = pData + strideBytes * y;
		for (uint32_t x = 0; x < width; ++x)
		{
			fprintf(f, "%d\n", pLine[x]);
		}
	}

	fclose(f);

	return true;
}

static void GenerateRandom(uint8_t* data, const uint32_t w, const uint32_t h)
{
	static std::random_device sRngSeed;
	static std::mt19937 sRng(sRngSeed());
	static std::uniform_int_distribution<uint32_t> sDist(0, 255);

	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			data[x + y * w] = static_cast<uint8_t>(sDist(sRng));
		}
	}
}

static void GenerateCheckerboard(uint8_t* data, const uint32_t w, const uint32_t h)
{
	static constexpr uint32_t patternSize = 256; //16;
	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			data[x + y * w] = (x / patternSize + y / patternSize) & 1u ? 0 : 255;
		}
	}
}

static void InvertCPU(
	uint8_t* data,
	const uint32_t n)
{
	for (uint32_t i = 0; i < n; ++i)
	{
		invert_kernel2(i, data);
	}
}

static void InvertGPU(
	MTL::CommandQueue* pCommandQueue,
	MTL::ComputePipelineState* pPSO,
	MTL::Buffer* data,
	const uint32_t n)
{
	// Enqueue GPU commands
	MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	// bind the compute pipeline
	pEncoder->setComputePipelineState(pPSO);
	pEncoder->setBuffer(data, 0, 0);

	NS::UInteger blockSize = pPSO->maxTotalThreadsPerThreadgroup();
	NS::UInteger blockCount = (n + blockSize - 1) / blockSize;
	pEncoder->dispatchThreadgroups(MTL::Size(blockCount, 1, 1), MTL::Size(blockSize, 1, 1));

	//MTL::Size gridSize = MTL::Size(n, 1, 1);
	//MTL::Size threadsPerThreadgroup = MTL::Size(pPSO->maxTotalThreadsPerThreadgroup(), 1, 1);
	//pEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);

	pEncoder->endEncoding();

	// execute computation on the GPU and wait for it
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
}


bool test4()
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
	NS::SharedPtr<MTL::Function> pInvertFn = TransferPtr(pLibrary->newFunction(NS::String::string("invert_kernel2", NS::UTF8StringEncoding)));
	NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(pDevice->newComputePipelineState(pInvertFn.get(), &pError));
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
	const size_t W = 8192;
	const size_t H = 8192;

	NS::SharedPtr<MTL::Buffer> resCPU = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resGPU = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));

	uint8_t* cpu = static_cast<uint8_t*>(resCPU->contents());
	uint8_t* gpu = static_cast<uint8_t*>(resGPU->contents());

	// Create input data
	std::vector<uint8_t> srcData(W * H);
	//GenerateRandom(static_cast<uint8_t*>(bufSrc->contents()), W, H);
	GenerateCheckerboard(srcData.data(), W, H);
	std::memcpy(cpu, srcData.data(), W * H * sizeof(uint8_t));
	std::memcpy(gpu, srcData.data(), W * H * sizeof(uint8_t));

	// write inputs as images
	//WritePPM("cpu_orig.ppm", cpu, W * sizeof(uint8_t), W, H);
	//WritePPM("gpu_orig.ppm", gpu, W * sizeof(uint8_t), W, H);

	// compute on CPU (reference)
	auto cpu_t0 = std::chrono::high_resolution_clock::now();
	InvertCPU(cpu, W * H);
	auto cpu_t1 = std::chrono::high_resolution_clock::now();

	// compute on GPU
	auto gpu_t0 = std::chrono::high_resolution_clock::now();
	InvertGPU(pCommandQueue.get(), pPSO.get(), resGPU.get(), W * H);
	auto gpu_t1 = std::chrono::high_resolution_clock::now();

	// write outputs as images
	//WritePPM("cpu.ppm", cpu, W * sizeof(uint8_t), W, H);
	//WritePPM("gpu.ppm", gpu, W * sizeof(uint8_t), W, H);

	// validate results
	for (uint32_t y = 0; y < H; ++y)
	{
		for (uint32_t x = 0; x < W; ++x)
		{
			const uint32_t i = x + y * W;
			if (cpu[i] != gpu[i])
			{
				std::cerr << "Invalid results, cpu and gpu differ at element [" << x << ", " << y << "]: " << (uint32_t)cpu[i] << " vs " << (uint32_t)gpu[i] << std::endl;
				return false;
			}
		}
	}

	std::cout << "CPU and GPU produces equivalent outputs" << std::endl;
	std::cout << "CPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_t1 - cpu_t0).count() << " ms" << std::endl;
	std::cout << "GPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(gpu_t1 - gpu_t0).count() << " ms" << std::endl;

	return true;
}
